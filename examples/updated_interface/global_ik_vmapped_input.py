import time

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from mujoco import viewer
from optax import OptState, ScaleByAdamState, adam
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier, PositionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.configuration import integrate
from mjinx.problem import Problem
from mjinx.solvers import GlobalIKData, GlobalIKSolver

# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mj_data = mj.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

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


problem.add_component(frame_task)
problem.add_component(position_barrier)
problem.add_component(joints_barrier)


# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = GlobalIKSolver(mjx_model, adam(learning_rate=1e-2), dt=1e-2)

N_batch = 200000
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
                -0.5,
                0.5,
                size=(mj_model.nq),
            ),
            q_min + 1e-1,
            q_max - 1e-1,
        )
        for _ in range(N_batch)
    ]
)

solver_data = solver.init(q)

# FIXME: I do not like the idea that you should know which function to vmap
solver.grad_fn = jax.jit(jax.vmap(solver.grad_fn, in_axes=(0, None)))
solve_jit = jax.jit(solver.solve)
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])


dt = 1e-2
ts = np.arange(0, 20, dt)

t_solve_avg = 0
n = 0

# Compiling
solve_jit(q, solver_data, problem_data)
print("~" * 100)

# Solution loop
for t in ts:
    # Changing desired values
    frame_task.target_frame = np.array([0.2 + 0.2 * jnp.sin(t) ** 2, 0.2, 0.2, 1, 0, 0, 0])
    # After changes, recompiling the model
    t0 = time.perf_counter()
    problem_data = problem.compile()

    # Solving the instance of the problem
    t1 = time.perf_counter()
    for _ in range(2):
        opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
    t2 = time.perf_counter()

    # Option 1:
    # q = integrate(mjx_model, q, opt_solution.v_opt, dt=dt)
    # Option 2:
    q = opt_solution.q_opt

    # MuJoCo visualization
    mj_data.qpos = q[0]
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

    t_solve = (t2 - t1) * 1e3
    # Ignore the first (compiling) iteration and calculate mean solution times
    if t > 0:
        t_solve_avg = t_solve_avg + (t_solve - t_solve_avg) / (n + 1)
        n += 1

print(f"Avg solving time: {t_solve_avg:0.3f}ms")
