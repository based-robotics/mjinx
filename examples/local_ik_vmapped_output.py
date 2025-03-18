"""
Example of Local inverse kinematics for a Kuka iiwa robot with vmapped output.

This example demonstrates how to use JAX's vmap to efficiently compute IK solutions
for multiple target poses. It shows how to set up the problem with batched outputs
and visualize the results using the BatchVisualizer.
"""

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from robot_descriptions.iiwa14_mj_description import MJCF_PATH
from time import perf_counter

from mjinx.components.barriers import JointBarrier, PositionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.configuration import integrate
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver
from mjinx.visualize import BatchVisualizer

print("=== Initializing ===")


# === Mujoco ===
print("Loading MuJoCo model...")
mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mj_data = mj.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.make_data(mjx_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

# --- Mujoco visualization ---
print("Setting up visualization...")
vis = BatchVisualizer(MJCF_PATH, n_models=4, alpha=0.5, record=False)

# Initialize a sphere marker for end-effector task
vis.add_markers(
    name=[f"ee_marker_{i}" for i in range(vis.n_models)],
    size=0.05,
    marker_alpha=0.5,
    color_begin=np.array([0, 1.0, 0.53]),
    color_end=np.array([0.38, 0.94, 1.0]),
    n_markers=vis.n_models,
)
vis.add_markers(
    name="blocking_plane",
    marker_type=mj.mjtGeom.mjGEOM_PLANE,
    size=np.array([0.5, 0.5, 0.02]),
    marker_alpha=0.7,
    color_begin=np.array([1, 0, 0]),
)

# === Mjinx ===
print("Setting up optimization problem...")
# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-5, v_max=5)

# Creating components of interest and adding them to the problem
frame_task = FrameTask("ee_task", cost=1, gain=20, obj_name="link7")
position_barrier = PositionBarrier(
    "ee_barrier",
    gain=100,
    obj_name="link7",
    limit_type="max",
    p_max=0.5,
    safe_displacement_gain=1e-2,
    mask=[1, 0, 0],
)
joints_barrier = JointBarrier("jnt_range", gain=10)
# Set plane coodinate same to limiting one

vis.marker_data["blocking_plane"].pos = np.array([position_barrier.p_max[0], 0, 0.3])
vis.marker_data["blocking_plane"].rot = np.array(
    [
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0],
    ]
)

problem.add_component(frame_task)
problem.add_component(position_barrier)
problem.add_component(joints_barrier)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
print("Initializing solver...")
solver = LocalIKSolver(mjx_model, maxiter=10)

# Initializing initial condition
N_batch = 65536
np.random.seed(42)
q0 = jnp.array(
    [
        -1.4238753,
        -1.7268502,
        -0.84355015,
        2.0962472,
        2.1339328,
        2.0837479,
        -2.5521986,
    ]
)
q = jnp.array(
    [
        np.clip(
            q0
            + np.random.uniform(
                -0.1,
                0.1,
                size=(mj_model.nq),
            ),
            q_min + 1e-1,
            q_max - 1e-1,
        )
        for _ in range(N_batch)
    ]
)

# --- Batching ---
print("Setting up batched computations...")
# First of all, data should be created via vmapped init function
solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

# To create a batch w.r.t. desired component's attributes, library defines convinient wrapper
# That sets all elements to None and allows user to mutate dataclasses of interest.
# After exiting the Context Manager, you'll get immutable jax dataclass object.
with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["ee_task"].target_frame = 0

# Vmapping solve and integrate functions.
solve_jit = jax.jit(jax.vmap(solver.solve, in_axes=(0, None, None, empty_problem_data)))
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])


def compute_target_frame(i, t):
    return jnp.array(
        [
            0.4 + 0.3 * jnp.sin(t + 2 * jnp.pi * i / N_batch),
            0.2,
            0.4 + 0.3 * jnp.cos(t + 2 * jnp.pi * i / N_batch),
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )


compute_target_frame_vmapped = jax.jit(jax.vmap(compute_target_frame, in_axes=(0, None)))
t_warmup = perf_counter()
print("Performing warmup calls...")
# Warmup iterations for JIT compilation
frame_task.target_frame = np.array([[0.4, 0.2, 0.7, 1, 0, 0, 0] for _ in range(N_batch)])
problem_data = problem.compile()
opt_solution, _ = solve_jit(q, mjx_data, solver_data, problem_data)
q_warmup = integrate_jit(mjx_model, q, opt_solution.v_opt, 0)
compute_target_frame_vmapped(jnp.arange(N_batch), 0)

t_warmup_duration = perf_counter() - t_warmup
print(f"Warmup completed in {t_warmup_duration:.3f} seconds")

# === Control loop ===
print("\n=== Starting main loop ===")
dt = 2e-2
ts = np.arange(0, 20, dt)

# Performance tracking
compile_times = []
solve_times = []
integrate_times = []
n_steps = 0

try:
    for t in ts:
        # Changing desired values
        i = jnp.arange(N_batch)
        frame_task.target_frame = compute_target_frame_vmapped(i, t)
        t1 = perf_counter()
        problem_data = problem.compile()
        t2 = perf_counter()
        compile_times.append(t2 - t1)

        # Solving the instance of the problem
        t1 = perf_counter()
        opt_solution, solver_data = solve_jit(q, mjx_data, solver_data, problem_data)
        t2 = perf_counter()
        solve_times.append(t2 - t1)

        # Integrating
        t1 = perf_counter()
        q = integrate_jit(
            mjx_model,
            q,
            opt_solution.v_opt,
            dt,
        )
        t2 = perf_counter()
        integrate_times.append(t2 - t1)

        # --- MuJoCo visualization ---
        for i, q_i in enumerate(frame_task.target_frame.wxyz_xyz[:: N_batch // vis.n_models, -3:]):
            vis.marker_data[f"ee_marker_{i}"].pos = q_i
        vis.update(q[:: N_batch // vis.n_models])
        n_steps += 1

except KeyboardInterrupt:
    print("\nSimulation interrupted by user")
except Exception as e:
    print(f"\nError occurred: {e}")
finally:
    if vis.record:
        vis.save_video(round(1 / dt))
    vis.close()

    # Print performance report
    print("\n=== Performance Report ===")
    print(f"Total steps completed: {n_steps}")
    print("\nComputation times per step:")
    if compile_times:
        avg_compile = sum(compile_times) / len(compile_times)
        std_compile = np.std(compile_times)
        print(f"compile        : {avg_compile * 1000:8.3f} ± {std_compile * 1000:8.3f} ms")
    if solve_times:
        avg_solve = sum(solve_times) / len(solve_times)
        std_solve = np.std(solve_times)
        print(f"solve          : {avg_solve * 1000:8.3f} ± {std_solve * 1000:8.3f} ms")
    if integrate_times:
        avg_integrate = sum(integrate_times) / len(integrate_times)
        std_integrate = np.std(integrate_times)
        print(f"integrate      : {avg_integrate * 1000:8.3f} ± {std_integrate * 1000:8.3f} ms")

    if solve_times and integrate_times:
        avg_total = sum(t1 + t2 + t3 for t1, t2, t3 in zip(compile_times, solve_times, integrate_times)) / len(
            solve_times
        )
        print(f"\nAverage computation time per step: {avg_total * 1000:.3f} ms")
        print(f"Effective computation rate: {1 / avg_total:.1f} Hz")
