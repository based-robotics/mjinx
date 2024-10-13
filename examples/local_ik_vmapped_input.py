import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier, PositionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.configuration import integrate
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver
from mjinx.visualize import BatchVisualizer

# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

# --- Mujoco visualization ---
# Initialize render window and launch it at the background
vis = BatchVisualizer(MJCF_PATH, n_models=5, alpha=0.5, record=False)

# Initialize a sphere marker for end-effector task
vis.add_markers(
    name="ee_marker",
    size=0.05,
    marker_alpha=0.4,
    color_begin=np.array([0, 1.0, 0.53]),
)
vis.add_markers(
    name="blocking_plane",
    marker_type=mj.mjtGeom.mjGEOM_PLANE,
    size=np.array([0.5, 0.5, 0.02]),
    marker_alpha=0.7,
    color_begin=np.array([1, 0, 0]),
)
# === Mjinx ===

# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-5, v_max=5)

# Creating components of interest and adding them to the problem
frame_task = FrameTask("ee_task", cost=1, gain=20, body_name="link7")
position_barrier = PositionBarrier(
    "ee_barrier",
    gain=100,
    body_name="link7",
    limit_type="max",
    p_max=0.4,
    safe_displacement_gain=1e-2,
    mask=[1, 0, 0],
)
# Set plane coodinate same to limiting one
vis.marker_data["blocking_plane"].pos = np.array([0.4, 0, 0.3])
vis.marker_data["blocking_plane"].rot = np.array(
    [
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0],
    ]
)
joints_barrier = JointBarrier("jnt_range", gain=10)

problem.add_component(frame_task)
problem.add_component(position_barrier)
problem.add_component(joints_barrier)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = LocalIKSolver(mjx_model, maxiter=20)

# Initializing initial condition
N_batch = 10000
q0 = np.array(
    [
        -1.5878328,
        -2.0968683,
        -1.4339591,
        1.6550868,
        2.1080072,
        1.646142,
        -2.982619,
    ]
)
q = jnp.array(
    [
        np.clip(
            q0.copy()
            + np.random.uniform(
                -1.0,
                1.0,
                size=(mj_model.nq),
            ),
            q_min,
            q_max,
        )
        for _ in range(N_batch)
    ]
)

# --- Batching ---
# First of all, data should be created via vmapped init function
solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

# Vmapping solve and integrate functions.
# Note that for batching w.r.t. q both q and solver_data should be batched.
# Other approaches might work, but it would be undefined behaviour, please stick to this format.
solve_jit = jax.jit(
    jax.vmap(
        solver.solve,
        in_axes=(0, 0, None),
    )
)
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])


# === Control loop ===
dt = 1e-2
ts = np.arange(0, 20.0, dt)

try:
    for t in ts:
        # Changing desired values
        frame_task.target_frame = np.array([0.4 + 0.3 * np.sin(t), 0.2, 0.4 + 0.3 * np.cos(t), 1, 0, 0, 0])
        # After changes, recompiling the model
        problem_data = problem.compile()

        # Solving the instance of the problem
        opt_solution, solver_data = solve_jit(q, solver_data, problem_data)

        # Integrating
        q = integrate_jit(
            mjx_model,
            q,
            opt_solution.v_opt,
            dt,
        )

        # --- MuJoCo visualization ---
        vis.marker_data["ee_marker"].pos = np.array(frame_task.target_frame.wxyz_xyz[-3:])
        vis.update(q[: vis.n_models])

except KeyboardInterrupt:
    print("Finalizing the simulation as requested...")
except Exception as e:
    print(e)
finally:
    if vis.record:
        vis.save_video(round(1 / dt))
    vis.close()
