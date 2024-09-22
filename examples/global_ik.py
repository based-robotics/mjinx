import time

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from mujoco import viewer
from optax import adam
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier, PositionBarrier, SelfCollisionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.configuration import integrate
from mjinx.problem import Problem
from mjinx.solvers import GlobalIKSolver

# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mj_data = mj.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()


# --- Mujoco visualization ---
# Initialize render window and launch it at the background
mj_data = mj.MjData(mj_model)
renderer = mj.Renderer(mj_model)
mj_viewer = viewer.launch_passive(
    mj_model,
    mj_data,
    show_left_ui=False,
    show_right_ui=False,
)

# Initialize a sphere marker for end-effector task
renderer.scene.ngeom += 1
mj_viewer.user_scn.ngeom = 1
mj.mjv_initGeom(
    mj_viewer.user_scn.geoms[0],
    mj.mjtGeom.mjGEOM_SPHERE,
    0.05 * np.ones(3),
    np.array([0.2, 0.2, 0.2]),
    np.eye(3).flatten(),
    np.array([0.565, 0.933, 0.565, 0.4]),
)

# === Mjinx ===

# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-100, v_max=100)

# Creating components of interest and adding them to the problem
frame_task = FrameTask("ee_task", cost=1, gain=50, body_name="link7")
position_barrier = PositionBarrier(
    "ee_barrier",
    gain=0.1,
    body_name="link7",
    limit_type="max",
    p_max=0.3,
    safe_displacement_gain=1e-2,
    mask=[1, 0, 0],
)
joints_barrier = JointBarrier("jnt_range", gain=0.1)
self_collision_barrier = SelfCollisionBarrier(
    "self_collision_barrier",
    0.01,
    d_min=0.01,
)

problem.add_component(frame_task)
problem.add_component(position_barrier)
problem.add_component(joints_barrier)
problem.add_component(self_collision_barrier)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = GlobalIKSolver(mjx_model, adam(learning_rate=1e-2), dt=1e-2)

# Initial condition
q = np.array(
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
solver_data = solver.init(q)

# Jit-compiling the key functions for better efficiency
solve_jit = jax.jit(solver.solve)
integrate_jit = jax.jit(integrate, static_argnames=["dt"])

# === Control loop ===
dt = 1e-2
ts = np.arange(0, 20, dt)

t_solve_avg = 0.0
n = 0

for t in ts:
    # Changing desired values
    frame_task.target_frame = np.array([0.2 + 0.2 * jnp.sin(t) ** 2, 0.2, 0.2, 1, 0, 0, 0])
    # After changes, recompiling the model
    t0 = time.perf_counter()
    problem_data = problem.compile()
    t1 = time.perf_counter()

    # Solving the instance of the problem
    for _ in range(1):
        opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
    t2 = time.perf_counter()

    # Two options for retriving q:
    # Option 1, integrating:
    # q = integrate(mjx_model, q, opt_solution.v_opt, dt=dt)
    # Option 2, direct:
    q = opt_solution.q_opt

    # --- MuJoCo visualization ---
    mj_data.qpos = q
    mj.mj_forward(mj_model, mj_data)
    print(f"Position barrier: {mj_data.xpos[position_barrier.body_id][0]} <= {position_barrier.p_max[0]}")
    mj.mjv_initGeom(
        mj_viewer.user_scn.geoms[0],
        mj.mjtGeom.mjGEOM_SPHERE,
        0.05 * np.ones(3),
        np.array(frame_task.target_frame.wxyz_xyz[-3:], dtype=np.float64),
        np.eye(3).flatten(),
        np.array([0.565, 0.933, 0.565, 0.4]),
    )

    # Run the forward dynamics to reflec
    # the updated state in the data
    mj.mj_forward(mj_model, mj_data)
    mj_viewer.sync()

    # --- Logging ---
    # Execution time
    t_solve = (t2 - t1) * 1e3
    # Ignore the first (compiling) iteration and calculate mean solution times
    if t > 0:
        t_solve_avg = t_solve_avg + (t_solve - t_solve_avg) / (n + 1)
        n += 1

print(f"Avg solving time: {t_solve_avg:0.3f}ms")
