import traceback
from time import perf_counter

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3
from robot_descriptions.cassie_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier
from mjinx.components.constraints import ModelEqualityConstraint
from mjinx.components.tasks import ComTask, FrameTask
from mjinx.configuration import integrate, update
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver
from mjinx.visualize import BatchVisualizer

# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mjx_model = mjx.put_model(mj_model)
print(mjx_model.nq, mjx_model.nv)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

# --- Mujoco visualization ---
# Initialize render window and launch it at the background
vis = BatchVisualizer(MJCF_PATH, n_models=5, alpha=0.2, record=True)

# === Mjinx ===
# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-1, v_max=1)

# Creating components of interest and adding them to the problem
joints_barrier = JointBarrier(
    "jnt_range",
    gain=0.1,
)
com_task = ComTask("com_task", cost=10.0, gain=50.0, mask=[1, 1, 1])
torso_task = FrameTask("torso_task", cost=1.0, gain=2.0, obj_name="cassie-pelvis", mask=[0, 0, 0, 1, 1, 1])

# Feet (in stance)
left_foot_task = FrameTask(
    "left_foot_task",
    cost=20.0,
    gain=10.0,
    obj_name="left-foot",
    mask=[1, 1, 1, 1, 0, 1],
)
right_foot_task = FrameTask(
    "right_foot_task",
    cost=20.0,
    gain=10.0,
    obj_name="right-foot",
    mask=[1, 1, 1, 1, 0, 1],
)

model_equality_constraint = ModelEqualityConstraint()

problem.add_component(com_task)
problem.add_component(torso_task)
# TODO: fix this
problem.add_component(joints_barrier)
problem.add_component(left_foot_task)
problem.add_component(right_foot_task)
problem.add_component(model_equality_constraint)
# Initializing solver and its initial state
solver = LocalIKSolver(mjx_model, maxiter=10)

# Initializing initial condition
N_batch = 100
q0 = mj_model.keyframe("home").qpos
q = jnp.tile(q0, (N_batch, 1))

mjx_data = update(mjx_model, jnp.array(q0))

com0 = np.array(mjx_data.subtree_com[mjx_model.body_rootid[0]])
com_task.target_com = com0
# Get torso orientation and set it as target
torso_id = mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_BODY, "cassie-pelvis")
torso_quat = mjx_data.xquat[torso_id]
torso_task.target_frame = np.concatenate([np.zeros(3), torso_quat])

left_foot_id = mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_BODY, "left-foot")
left_foot_pos = mjx_data.xpos[left_foot_id]
left_foot_quat = mjx_data.xquat[left_foot_id]
left_foot_task.target_frame = jnp.concatenate([left_foot_pos, left_foot_quat])

right_foot_id = mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_BODY, "right-foot")
right_foot_pos = mjx_data.xpos[right_foot_id]
right_foot_quat = mjx_data.xquat[right_foot_id]
right_foot_task.target_frame = jnp.concatenate([right_foot_pos, right_foot_quat])

# Compiling the problem upon any parameters update
problem_data = problem.compile()
# --- Batching ---
print("Setting up batched computations...")
solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["com_task"].target_com = 0

# Vmapping solve and integrate functions.
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

try:
    for t in ts:
        # Solving the instance of the problem
        # com_task.target_com = com0 - np.array([0, 0, 0.2 * np.sin(t) ** 2])
        com_task.target_com = np.array(
            [[0.0, 0.0, com0[2] - 0.3 * np.sin(t + 2 * np.pi * i / N_batch + np.pi / 2) ** 2] for i in range(N_batch)]
        )
        problem_data = problem.compile()
        opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
        # Integrating
        q = integrate_jit(
            mjx_model,
            q,
            opt_solution.v_opt,
            dt,
        )
        # --- MuJoCo visualization ---
        indices = np.arange(0, N_batch, N_batch // vis.n_models)

        vis.update(q[:: N_batch // vis.n_models])

except KeyboardInterrupt:
    print("Finalizing the simulation as requested...")
except Exception:
    print(traceback.format_exc())
finally:
    if vis.record:
        vis.save_video(round(1 / dt))
    vis.close()
