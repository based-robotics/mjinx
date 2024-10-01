import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from optax import adam
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier, PositionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.problem import Problem
from mjinx.solvers import GlobalIKSolver


def test_global_ik_jit():
    mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
    mjx_model = mjx.put_model(mj_model)

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

    problem.add_component(frame_task)
    problem.add_component(position_barrier)
    problem.add_component(joints_barrier)

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

    # === Control loop ===
    dt = 1e-2
    ts = np.arange(0, 1, dt)

    for t in ts:
        # Changing desired values
        frame_task.target_frame = np.array([0.2 + 0.2 * jnp.sin(t) ** 2, 0.2, 0.2, 1, 0, 0, 0])
        # After changes, recompiling the model
        problem_data = problem.compile()
        # Solving the instance of the problem
        for _ in range(1):
            opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
