import time

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

from mjinx.components.barriers import PositionBarrier
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
mj_model.jnt_range = [(-20 * np.pi, 20 * np.pi) for _ in range(7)]

# === Mjinx ===

# == Constructing the problem

# Creating problem formulation
problem = Problem(mjx_model, v_min=-1, v_max=1)

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

problem.add_component(frame_task)
problem.add_component(position_barrier)


# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = LocalIKSolver(mjx_model, maxiter=20)

q = jnp.array(
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
solver_data = solver.init(q0=q)
solve_jit = jax.jit(solver.solve)
integrate_jit = jax.jit(integrate, static_argnames=["dt"])

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
    v_opt, solver_data = solve_jit(q, solver_data, problem_data)
    t1 = time.perf_counter()
    # Integrating
    q = integrate_jit(
        mjx_model,
        q,
        velocity=v_opt,
        dt=dt,
    )
    t2 = time.perf_counter()
    t_solve = (t1 - t0) * 1e3
    t_interpolate = (t2 - t1) * 1e3

    t_solve_avg = t_solve_avg + (t_solve - t_solve_avg) / (n + 1)
    n += 1


print(f"Avg solving time: {t_solve_avg:0.3f}ms")
