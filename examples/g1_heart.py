import traceback
from time import perf_counter
import os.path
from collections import defaultdict

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3

from mjinx.components.barriers import JointBarrier, SelfCollisionBarrier
from mjinx.components.tasks import ComTask, FrameTask
from mjinx.configuration import integrate, update
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver
from mjinx.visualize import BatchVisualizer

print("=== Initializing ===")
# t_start = perf_counter()

# === Mujoco ===
print("Loading MuJoCo model...")
# Define workspace root for absolute paths
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(WORKSPACE_ROOT, "examples/g1_description/g1.xml")

mj_model = mj.MjModel.from_xml_path(MODEL_PATH)
mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

# --- Mujoco visualization ---
print("Setting up visualization...")
# Initialize render window and launch it at the background
vis = BatchVisualizer(MODEL_PATH, n_models=10, alpha=0.1, record=True)
vis.add_markers(
    name=[f"left_arm_{i}" for i in range(vis.n_models)],
    size=0.035,
    marker_alpha=0.8,
    color_begin=np.array([1.0, 0.0, 0.0]),
    color_end=np.array([1.0, 0.0, 0.0]),
    n_markers=vis.n_models,
)

vis.add_markers(
    name=[f"right_arm_{i}" for i in range(vis.n_models)],
    size=0.035,
    marker_alpha=0.8,
    color_begin=np.array([1.0, 0.0, 0.0]),
    color_end=np.array([1.0, 0.0, 0.0]),
    n_markers=vis.n_models,
)

# === Mjinx ===
print("Setting up optimization problem...")
# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-5, v_max=5)

# Creating components of interest and adding them to the problem
joints_barrier = JointBarrier("jnt_range", gain=1.0, floating_base=True)

com_task = ComTask("com_task", cost=1.0, gain=2.0, mask=[1, 1, 0])
torso_task = FrameTask("torso_task", cost=1.0, gain=2.0, obj_name="pelvis", mask=[0, 0, 0, 1, 1, 1])

# Arms (moving)
left_arm_task = FrameTask(
    "left_arm_task",
    cost=1.0,
    gain=20.0,
    obj_type=mj.mjtObj.mjOBJ_BODY,
    obj_name="left_one_link",
    mask=[1, 1, 1, 1, 1, 0],
)
right_arm_task = FrameTask(
    "right_arm_task",
    cost=1.0,
    gain=20.0,
    obj_type=mj.mjtObj.mjOBJ_BODY,
    obj_name="right_one_link",
    mask=[1, 1, 1, 1, 1, 0],
)

# Feet (in stance)
left_foot_task = FrameTask(
    "left_foot_task",
    cost=5.0,
    gain=50.0,
    obj_type=mj.mjtObj.mjOBJ_SITE,
    obj_name="left_foot",
)
right_foot_task = FrameTask(
    "right_foot_task",
    cost=5.0,
    gain=50.0,
    obj_type=mj.mjtObj.mjOBJ_SITE,
    obj_name="right_foot",
)

# Avoiding collision between arms and torso
self_collision_barrier = SelfCollisionBarrier(
    "self_collision_barrier",
    gain=50.0,
    safe_displacement_gain=1e-2,
    collision_bodies=[
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_roll_link",
        "right_shoulder_roll_link",
        "right_elbow_roll_link",
    ],
    excluded_collisions=[
        ("left_shoulder_roll_link", "left_elbow_roll_link"),
        ("right_shoulder_roll_link", "right_elbow_roll_link"),
    ],
)

problem.add_component(com_task)
problem.add_component(torso_task)
problem.add_component(joints_barrier)
problem.add_component(left_arm_task)
problem.add_component(right_arm_task)
problem.add_component(left_foot_task)
problem.add_component(right_foot_task)
problem.add_component(self_collision_barrier)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
print("Initializing solver...")
solver = LocalIKSolver(mjx_model, maxiter=10)

# Initializing initial condition
N_batch = 500
q0 = mj_model.keyframe("stand").qpos
q = jnp.tile(q0, (N_batch, 1))

# TODO: implement update_from_model_data
mjx_data = update(mjx_model, jnp.array(q0))

com_pos = mjx_data.subtree_com[mjx_model.body_rootid[0]]
com_task.target_com = com_pos[:2]

torso_task.target_frame = np.array([0, 0, 0, 1, 0, 0, 0])

left_foot_pos = mjx_data.site_xpos[mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_SITE, "left_foot")]
left_foot_task.target_frame = jnp.array([*left_foot_pos, 1, 0, 0, 0])

right_foot_pos = mjx_data.site_xpos[mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_SITE, "right_foot")]
right_foot_task.target_frame = jnp.array([*right_foot_pos, 1, 0, 0, 0])

# --- Batching ---
print("Setting up batched computations...")
solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["left_arm_task"].target_frame = 0
    empty_problem_data.components["right_arm_task"].target_frame = 0

solve_jit = jax.jit(
    jax.vmap(
        solver.solve,
        in_axes=(0, 0, empty_problem_data),
    )
)
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])

t_warmup = perf_counter()
print("Performing warmup calls...")
# Warmup iterations for JIT compilation
p0 = np.array([0.25, 0.0, 0.9])
right_series = np.linspace(0, np.pi, N_batch)
left_series = 2 * np.pi - right_series

def heart_curve(t: np.ndarray, p0: np.ndarray | None = None) -> np.ndarray:
    """Heart-like function.

    Function is shifted so that heart_curve(0) = p0.

    :param t: time, a.k.a phase
    :param p0: initial heart shift, defaults to zero
    :return: point on the heart contour
    :ref: https://pavpanchekha.com/blog/heart-polar-coordinates.html
    """
    # Shift the phase
    t += np.pi / 2

    # Polar coordinates formula
    r = 0.1 * (np.sin(t) * np.sqrt(abs(np.cos(t))) / (np.sin(t) + 7 / 5) - 2 * np.sin(t) + 2)
    # Cartesian coordinates formula
    pts = np.array([np.zeros_like(t), r * np.cos(t), r * np.sin(t)])

    # Shift the cartesian point w.r.t. initial point
    p0 = p0 if p0 is not None else np.zeros(3)

    return np.tile(p0, (len(t), 1)) + pts.T


def triangle_wave(t: np.ndarray) -> np.ndarray:
    """Triangle wave, amplitude is from 0 to pi, and frequency pi

    :param t: input variable
    :return: output variable
    """
    return np.where(t % (2 * np.pi) < np.pi, t % np.pi, np.pi - (t % np.pi))



left_arm_task.target_frame = np.concatenate(
    (
        heart_curve(left_series, p0),
        np.tile(np.array((1, 0, 0, 0)), (N_batch, 1)),
    ),
    axis=1,
)
right_arm_task.target_frame = np.concatenate(
    (
        heart_curve(right_series, p0),
        np.tile(np.array((1, 0, 0, 0)), (N_batch, 1)),
    ),
    axis=1,
)
problem_data = problem.compile()
opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
q_warmup = integrate_jit(mjx_model, q, opt_solution.v_opt, 1e-2)

t_warmup_duration = perf_counter() - t_warmup
print(f"Warmup completed in {t_warmup_duration:.3f} seconds")

# === Control loop ===
print("\n=== Starting main loop ===")
dt = 2e-2
ts = np.arange(0, 10, dt)



p0 = np.array([0.25, 0.0, 0.9])

# Performance tracking
solve_times = []
integrate_times = []
n_steps = 0

try:
    for t in ts:
        # Changing desired values
        right_series = triangle_wave(np.linspace(t, np.pi + t, N_batch))
        left_series = 2 * np.pi - right_series
        left_arm_task.target_frame = np.concatenate(
            (
                heart_curve(left_series, p0),
                np.tile(np.array((1, 0, 0, 0)), (N_batch, 1)),
            ),
            axis=1,
        )
        right_arm_task.target_frame = np.concatenate(
            (
                heart_curve(right_series, p0),
                np.tile(np.array((1, 0, 0, 0)), (N_batch, 1)),
            ),
            axis=1,
        )

        # After changes, recompiling the model
        problem_data = problem.compile()

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
        indices = np.arange(0, N_batch, N_batch // vis.n_models)
        left_arm_viz = left_arm_task.target_frame.translation()[indices]
        right_arm_viz = right_arm_task.target_frame.translation()[indices]
        for i in range(vis.n_models):
            vis.marker_data[f"left_arm_{i}"].pos = left_arm_viz[i]
            vis.marker_data[f"right_arm_{i}"].pos = right_arm_viz[i]

        vis.update(q[indices])
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
    if solve_times:
        avg_solve = sum(solve_times) / len(solve_times)
        std_solve = np.std(solve_times)
        print(f"solve          : {avg_solve*1000:8.3f} ± {std_solve*1000:8.3f} ms")
    if integrate_times:
        avg_integrate = sum(integrate_times) / len(integrate_times)
        std_integrate = np.std(integrate_times)
        print(f"integrate      : {avg_integrate*1000:8.3f} ± {std_integrate*1000:8.3f} ms")
    
    if solve_times and integrate_times:
        avg_total = sum(t1 + t2 for t1, t2 in zip(solve_times, integrate_times)) / len(solve_times)
        print(f"\nAverage computation time per step: {avg_total*1000:.3f} ms")
        print(f"Effective computation rate: {1/avg_total:.1f} Hz")
