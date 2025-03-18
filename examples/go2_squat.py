"""
Example of batched squats for a Go2 quadruped robot.

This example demonstrates how to use the mjinx library to generate squatting motions
for a Go2 quadruped robot. It shows how to set up the problem with multiple tasks
for different parts of the robot, and visualize the results using the BatchVisualizer.
"""

import traceback
from os.path import join
from time import perf_counter
from collections import defaultdict

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3
from robot_descriptions.go2_mj_description import PACKAGE_PATH

from mjinx.components.barriers import JointBarrier
from mjinx.components.tasks import ComTask, FrameTask
from mjinx.configuration import integrate, update
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver
from mjinx.visualize import BatchVisualizer

print("=== Initializing ===")


MJCF_PATH = join(PACKAGE_PATH, "go2_mjx.xml")
# === Mujoco ===
print("Loading MuJoCo model...")
mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

# --- Mujoco visualization ---
print("Setting up visualization...")
# Initialize render window and launch it at the background
vis = BatchVisualizer(MJCF_PATH, n_models=10, alpha=0.2)

# === Mjinx ===
print("Setting up optimization problem...")
# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-5, v_max=5)

# Creating components of interest and adding them to the problem
com_task = ComTask("com_task", cost=5.0, gain=10.0)
frame_task = FrameTask("body_orientation_task", cost=1.0, gain=10, obj_name="base", mask=[0, 0, 0, 1, 1, 1])
joints_barrier = JointBarrier("jnt_range", gain=0.1)

problem.add_component(com_task)
problem.add_component(frame_task)
problem.add_component(joints_barrier)

for foot in ["FL", "FR", "RL", "RR"]:
    task = FrameTask(
        foot + "_foot_task",
        obj_name=foot,
        obj_type=mj.mjtObj.mjOBJ_GEOM,
        cost=20.0,
        gain=10.0,
        mask=[1, 1, 1, 0, 0, 0],
    )
    problem.add_component(task)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
print("Initializing solver...")
solver = LocalIKSolver(mjx_model, maxiter=10)

# Initializing initial condition
N_batch = 10000
q0 = np.array(
    [
        -0.0,  # Torso position
        0.0,
        0.3,
        1.0,  # Torso orientation
        0.0,
        0.0,
        0.0,
        0.0,  # Front Left foot
        0.8,
        -1.57,
        0.0,  # Front Right foot
        0.8,
        -1.57,
        0.0,  # Rare Left foot
        0.8,
        -1.57,
        0.0,  # Rare Right foot
        0.8,
        -1.57,
    ]
)
q = jnp.tile(q0, (N_batch, 1))

# TODO: implement update_from_model_data
mjx_data = update(mjx_model, jnp.array(q0))

for foot in ["FL", "FR", "RL", "RR"]:
    foot_pos = mjx_data.geom_xpos[mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_GEOM, foot)]
    problem.component(foot + "_foot_task").target_frame = jnp.array([*foot_pos, 1, 0, 0, 0])

# --- Batching ---
print("Setting up batched computations...")
# First of all, data should be created via vmapped init function
solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

# To create a batch w.r.t. desired component's attributes, library defines convinient wrapper
# That sets all elements to None and allows user to mutate dataclasses of interest.
# After exiting the Context Manager, you'll get immutable jax dataclass object.
with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["body_orientation_task"].target_frame = 0
    empty_problem_data.components["com_task"].target_com = 0

# Vmapping solve and integrate functions.
solve_jit = jax.jit(
    jax.vmap(
        solver.solve,
        in_axes=(0, 0, empty_problem_data),
    )
)
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])


def get_frame_traj(i: int, t: float) -> np.ndarray:
    angle = np.pi / 8 * np.sin(t + 2 * np.pi * i / N_batch)
    R = jnp.array(
        [
            [jnp.cos(angle), 0, jnp.sin(angle)],
            [0, 1, 0],
            [-jnp.sin(angle), 0, jnp.cos(angle)],
        ]
    )
    return SE3.from_rotation_and_translation(
        SO3.from_matrix(R),
        np.zeros(3),
    )


def get_com_traj(i: int, t: float) -> np.ndarray:
    return jnp.array([0.0, 0.0, 0.2 + 0.1 * jnp.sin(t + 2 * jnp.pi * i / N_batch + jnp.pi / 2)])


get_frame_traj_vmapped = jax.jit(jax.vmap(get_frame_traj, in_axes=(0, None)))
get_com_traj_vmapped = jax.jit(jax.vmap(get_com_traj, in_axes=(0, None)))

t_warmup = perf_counter()
print("Performing warmup calls...")
# Warmup iterations for JIT compilation
frame_task.target_frame = SE3.from_rotation_and_translation(
    SO3.from_matrix(np.stack([np.eye(3) for _ in range(N_batch)], axis=0)),
    np.zeros((N_batch, 3)),
)
com_task.target_com = np.array([[0.0, 0.0, 0.2] for _ in range(N_batch)])
problem_data = problem.compile()
opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
q_warmup = integrate_jit(mjx_model, q, opt_solution.v_opt, 1e-2)

get_frame_traj_vmapped(jnp.arange(N_batch), 0.0)
get_com_traj_vmapped(jnp.arange(N_batch), 0.0)

t_warmup_duration = perf_counter() - t_warmup
print(f"Warmup completed in {t_warmup_duration:.3f} seconds")

# === Control loop ===
print("\n=== Starting main loop ===")
dt = 1e-2
ts = np.arange(0, 20, dt)

# Performance tracking
compile_times = []
solve_times = []
integrate_times = []
n_steps = 0

try:
    for t in ts:
        # Changing desired values
        frame_task.target_frame = get_frame_traj_vmapped(jnp.arange(N_batch), t)
        com_task.target_com = get_com_traj_vmapped(jnp.arange(N_batch), t)

        # After changes, recompiling the model
        t1 = perf_counter()
        problem_data = problem.compile()
        t2 = perf_counter()
        compile_times.append(t2 - t1)

        # Solving the instance of the problem
        t1 = perf_counter()
        opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
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
        vis.update(q[:: N_batch // vis.n_models])
        n_steps += 1

except KeyboardInterrupt:
    print("\nSimulation interrupted by user")
except Exception:
    print(traceback.format_exc())
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
