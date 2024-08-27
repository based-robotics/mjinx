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
print(MJCF_PATH)
mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mj_data = mj.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)


# === Mjinx ===

# == Constructing the problem

# Creating problem formulation
problem = Problem(mjx_model)

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
solver = LocalIKSolver(mjx_model, maxiter=500)

q = jnp.zeros(7)
solver_data = solver.init(q0=q)

# Solution loop
while True:
    # Changing desired values
    frame_task.target_frame = np.array([0, 0, 0, 1, 0, 0, 0])

    # After changes, recompiling the model
    problem_data = problem.compile()

    # Solving the instance of the problem
    v_opt, solver_data = solver.solve(q, solver_data, problem_data)

    # Integrating
    q = integrate(
        mjx_model,
        mjx.make_data(mjx_model).replace(qpos=q),  # TODO: make integrate function from q
        velocity=v_opt,
        dt=1e-3,
    )
