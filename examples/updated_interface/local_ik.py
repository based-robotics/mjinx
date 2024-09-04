import time

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from mujoco import viewer
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier, PositionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.configuration import integrate
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver

# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mj_data = mj.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()
print(q_min, q_max)
# --- Mujoco visualization ---
mj_data = mj.MjData(mj_model)
renderer = mj.Renderer(mj_model)
mj_viewer = viewer.launch_passive(
    mj_model,
    mj_data,
    show_left_ui=False,
    show_right_ui=False,
)
renderer.scene.ngeom += 1
mj_viewer.user_scn.ngeom = 1
mj.mjv_initGeom(
    mj_viewer.user_scn.geoms[0],
    mj.mjtGeom.mjGEOM_SPHERE,
    0.1 * np.ones(3),
    np.array([0.2, 0.2, 0.2]),
    np.eye(3).flatten(),
    np.array([0.565, 0.933, 0.565, 0.4]),
)

# === Mjinx ===

# --- Constructing the problem ---

# Creating problem formulation
problem = Problem(mjx_model, v_min=-100, v_max=100)

frame_task = FrameTask("ee_task", cost=1, gain=20, body_name="link7")
position_barrier = PositionBarrier(
    "ee_barrier",
    gain=100,
    body_name="link7",
    limit_type="max",
    p_max=0.3,
    safe_displacement_gain=1e-2,
    mask=[1, 0, 0],
)
joints_barrier = JointBarrier("jnt_range", gain=10)
# mj_model.jnt_range = [(-20 * np.pi, 20 * np.pi) for _ in range(7)]


problem.add_component(frame_task)
problem.add_component(position_barrier)
problem.add_component(joints_barrier)


# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = LocalIKSolver(mjx_model, maxiter=20)

q = jnp.array(
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
solver_data = solver.init()

solve_jit = jax.jit(solver.solve)
integrate_jit = jax.jit(integrate, static_argnames=["dt"])
# solve_jit = solver.solve
# integrate_jit = integrate


dt = 1e-2
ts = np.arange(0, 20, dt)

t_solve_avg = 0
n = 0

# Solution loop
for t in ts:
    # Changing desired values
    frame_task.target_frame = np.array([0.2 + 0.2 * jnp.sin(t) ** 2, 0.2, 0.2, 1, 0, 0, 0])
    # After changes, recompiling the model
    problem_data = problem.compile()

    # Solving the instance of the problem
    t0 = time.perf_counter()
    solver_solution, solver_data = solve_jit(q, solver_data, problem_data)
    t1 = time.perf_counter()
    # Integrating
    q = integrate_jit(
        mjx_model,
        q,
        velocity=solver_solution.v_opt,
        dt=dt,
    )

    # MuJoCo visualization
    mj_data.qpos = q
    mj.mj_forward(mj_model, mj_data)
    print(f"Position barrier: {mj_data.xpos[position_barrier.body_id][0]} <= {position_barrier.p_max[0]}")
    mj.mjv_initGeom(
        mj_viewer.user_scn.geoms[0],
        mj.mjtGeom.mjGEOM_SPHERE,
        0.1 * np.ones(3),
        np.array(frame_task.target_frame.wxyz_xyz[-3:], dtype=np.float64),
        np.eye(3).flatten(),
        np.array([0.565, 0.933, 0.565, 0.4]),
    )

    # Run the forward dynamics to reflec
    # the updated state in the data
    mj.mj_forward(mj_model, mj_data)
    mj_viewer.sync()

    t2 = time.perf_counter()
    t_solve = (t1 - t0) * 1e3
    t_interpolate = (t2 - t1) * 1e3

    if t > 0:
        t_solve_avg = t_solve_avg + (t_solve - t_solve_avg) / (n + 1)
        n += 1
    # print(q)
    # print(v_opt)
    # print("-" * 100)

print(f"Avg solving time: {t_solve_avg:0.3f}ms")
