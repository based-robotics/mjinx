import traceback
from os.path import join

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3
from robot_descriptions.go2_mj_description import PACKAGE_PATH

from mjinx.components.barriers import JointBarrier
from mjinx.components.tasks import ComTask, FrameTask
from mjinx.configuration import integrate, update
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver
from mjinx.visualize import BatchVisualizer

MJCF_PATH = join(PACKAGE_PATH, "go2_mjx.xml")
# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

# --- Mujoco visualization ---
# Initialize render window and launch it at the background
vis = BatchVisualizer(MJCF_PATH, n_models=10, alpha=0.2)

# === Mjinx ===
# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-5, v_max=5)

# Creating components of interest and adding them to the problem
com_task = ComTask("com_task", cost=5.0, gain=10.0)
frame_task = FrameTask("body_orientation_task", cost=1.0, gain=10, obj_name="base", mask=[0, 0, 0, 1, 1, 1])
joints_barrier = JointBarrier("jnt_range", gain=0.1, floating_base=True)

problem.add_component(com_task)
problem.add_component(frame_task)
problem.add_component(joints_barrier)

for foot in ["FL", "FR", "RL", "RR"]:
    task = FrameTask(
        foot + "_foot_task",
        obj_name=foot,
        obj_type=mj.mjtObj.mjOBJ_GEOM,
        cost=20.0,
        gain=10.0,
        mask=[1, 1, 1, 0, 0, 0],
    )
    problem.add_component(task)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = LocalIKSolver(mjx_model, maxiter=10)

# Initializing initial condition
N_batch = 10000
q0 = np.array(
    [
        -0.0,  # Torso position
        0.0,
        0.3,
        1.0,  # Torso orientation
        0.0,
        0.0,
        0.0,
        0.0,  # Front Left foot
        0.8,
        -1.57,
        0.0,  # Front Right foot
        0.8,
        -1.57,
        0.0,  # Rare Left foot
        0.8,
        -1.57,
        0.0,  # Rare Right foot
        0.8,
        -1.57,
    ]
)
q = jnp.tile(q0, (N_batch, 1))

# TODO: implement update_from_model_data
mjx_data = update(mjx_model, jnp.array(q0))

for foot in ["FL", "FR", "RL", "RR"]:
    foot_pos = mjx_data.geom_xpos[mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_GEOM, foot)]
    problem.component(foot + "_foot_task").target_frame = jnp.array([*foot_pos, 1, 0, 0, 0])

# --- Batching ---
# First of all, data should be created via vmapped init function
solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

# To create a batch w.r.t. desired component's attributes, library defines convinient wrapper
# That sets all elements to None and allows user to mutate dataclasses of interest.
# After exiting the Context Manager, you'll get immutable jax dataclass object.
with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["body_orientation_task"].target_frame = 0
    empty_problem_data.components["com_task"].target_com = 0

# Vmapping solve and integrate functions.
# Note that for batching w.r.t. q both q and solver_data should be batched.
# Other approaches might work, but it would be undefined behaviour, please stick to this format.
solve_jit = jax.jit(
    jax.vmap(
        solver.solve,
        in_axes=(0, 0, empty_problem_data),
    )
)
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])

# === Control loop ===
dt = 1e-2
ts = np.arange(0, 20, dt)


def get_rotation(t: float, idx: int) -> np.ndarray:
    angle = np.pi / 8 * np.sin(t + 2 * np.pi * idx / N_batch)
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


try:
    for t in ts:
        # Changing desired values
        frame_task.target_frame = SE3.from_rotation_and_translation(
            SO3.from_matrix(
                np.stack(
                    [get_rotation(t, i) for i in range(N_batch)],
                    axis=0,
                ),
            ),
            np.zeros((N_batch, 3)),
        )
        com_task.target_com = np.array(
            [[0.0, 0.0, 0.2 + 0.1 * np.sin(t + 2 * np.pi * i / N_batch + np.pi / 2)] for i in range(N_batch)]
        )
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
        vis.update(q[:: N_batch // vis.n_models])

except KeyboardInterrupt:
    print("Finalizing the simulation as requested...")
except Exception:
    print(traceback.format_exc())
finally:
    if vis.record:
        vis.save_video(round(1 / dt))
    vis.close()
