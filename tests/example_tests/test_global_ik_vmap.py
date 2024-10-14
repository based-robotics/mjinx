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

    q_min = mj_model.jnt_range[:, 0].copy()
    q_max = mj_model.jnt_range[:, 1].copy()

    # === Mjinx ===

    # --- Constructing the problem ---
    # Creating problem formulation
    problem = Problem(mjx_model, v_min=-100, v_max=100)

    # Creating components of interest and adding them to the problem
    frame_task = FrameTask("ee_task", cost=1, gain=50, obj_name="link7")
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

    # Initializing initial condition
    N_batch = 1000
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

    # --- Batching ---
    # First of all, data should be created via vmapped init function
    solver_data = jax.vmap(solver.init, in_axes=0)(q)

    # To create a batch w.r.t. desired component's attributes, library defines convinient wrapper
    # That sets all elements to None and allows user to mutate dataclasses of interest.
    # After exiting the Context Manager, you'll get immutable jax dataclass object.
    with problem.set_vmap_dimension() as empty_problem_data:
        empty_problem_data.components["ee_task"].target_frame = 0

    # Vmapping solve and integrate functions.
    # Note that for batching w.r.t. q both q and solver_data should be batched.
    # Other approaches might work, but it would be undefined behaviour, please stick to this format.
    solve_jit = jax.jit(jax.vmap(solver.solve, in_axes=(0, 0, empty_problem_data)))

    # === Control loop ===
    dt = 1e-2
    ts = np.arange(0, 1, dt)

    for t in ts:
        # Changing desired values
        frame_task.target_frame = np.array(
            [[0.2 + 0.2 * np.sin(t + np.pi * i / N_batch) ** 2, 0.2, 0.2, 1, 0, 0, 0] for i in range(N_batch)]
        )
        # After changes, recompiling the model
        problem_data = problem.compile()
        # Solving the instance of the problem
        for _ in range(1):
            opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
