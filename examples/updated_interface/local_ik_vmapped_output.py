import time

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3
from mujoco import viewer
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier, PositionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.components.tasks._body_frame_task import JaxFrameTask
from mjinx.configuration import integrate
from mjinx.problem import JaxProblemData, Problem
from mjinx.solvers import LocalIKData, LocalIKSolver

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
    axes="x",
    p_max=0.3,
    safe_displacement_gain=1e-2,
)
joints_barrier = JointBarrier("jnt_range", gain=10)

problem.add_component(frame_task)
problem.add_component(position_barrier)
problem.add_component(joints_barrier)


# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = LocalIKSolver(mjx_model, maxiter=20)

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
q = jnp.array([q0.copy() for _ in range(N_batch)])
solver_data = solver.init(q=q)

# Batch the problem along desired frame in the task
with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["ee_task"].target_frame = 0

solve_jit = jax.jit(
    jax.vmap(
        solver.solve,
        in_axes=(0, empty_problem_data, None),
    )
)
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])

dt = 1e-2
ts = np.arange(0, 20, dt)

t_solve_avg = 0
n = 0

# Solution loop
for t in ts:
    # Changing desired values
    frame_task.target_frame = np.array(
        [[0.2 + 0.2 * np.sin(t + np.pi * i / N_batch) ** 2, 0.2, 0.2, 1, 0, 0, 0] for i in range(N_batch)]
    )
    # After changes, recompiling the model
    problem_data = problem.compile()
    # Solving the instance of the problem
    t0 = time.perf_counter()
    v_opt, solver_data = solve_jit(q, problem_data, solver_data)
    t1 = time.perf_counter()
    # Integrating
    q = integrate_jit(mjx_model, q, v_opt, dt)

    # MuJoCo visualization
    mj_data.qpos = q[0]
    mj.mj_forward(mj_model, mj_data)
    mj.mjv_initGeom(
        mj_viewer.user_scn.geoms[0],
        mj.mjtGeom.mjGEOM_SPHERE,
        0.1 * np.ones(3),
        np.array(frame_task.target_frame.wxyz_xyz[0, -3:], dtype=np.float64),
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

print(f"Avg solving time: {t_solve_avg:0.3f}ms")
