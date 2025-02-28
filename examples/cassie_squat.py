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
vis = BatchVisualizer(MJCF_PATH, n_models=1, alpha=1, record=False)

# === Mjinx ===
# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-5, v_max=5)

# Creating components of interest and adding them to the problem
joints_barrier = JointBarrier("jnt_range", gain=1.0, floating_base=True)

com_task = ComTask("com_task", cost=1.0, gain=2.0, mask=[1, 1, 1])
torso_task = FrameTask("torso_task", cost=1.0, gain=2.0, obj_name="cassie-pelvis", mask=[0, 0, 0, 1, 1, 1])

# Feet (in stance)
left_foot_task = FrameTask(
    "left_foot_task",
    cost=5.0,
    gain=50.0,
    obj_type=mj.mjtObj.mjOBJ_BODY,
    obj_name="left-foot",
)
right_foot_task = FrameTask(
    "right_foot_task",
    cost=5.0,
    gain=50.0,
    obj_type=mj.mjtObj.mjOBJ_BODY,
    obj_name="right-foot",
)

model_equality_constraint = ModelEqualityConstraint("model_eq_constraint", gain=1.0)

problem.add_component(com_task)
problem.add_component(torso_task)
problem.add_component(joints_barrier)
problem.add_component(left_foot_task)
problem.add_component(right_foot_task)
# problem.add_component(model_equality_constraint)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = LocalIKSolver(mjx_model, maxiter=10)

# Initializing initial condition
N_batch = 1
q0 = mj_model.keyframe("home").qpos
q = jnp.tile(q0, (N_batch, 1))

mjx_data = update(mjx_model, jnp.array(q0))

com0 = np.array(mjx_data.subtree_com[mjx_model.body_rootid[0]])
torso_task.target_frame = np.array([0, 0, 0, 1, 0, 0, 0])

left_foot_pos = mjx_data.site_xpos[mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_BODY, "left_foot")]
left_foot_task.target_frame = jnp.array([*left_foot_pos, 1, 0, 0, 0])

right_foot_pos = mjx_data.site_xpos[mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_BODY, "right_foot")]
right_foot_task.target_frame = jnp.array([*right_foot_pos, 1, 0, 0, 0])

# --- Batching ---
solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

solve_jit = jax.jit(
    jax.vmap(
        solver.solve,
        in_axes=(0, 0, None),
    )
)
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])

# === Control loop ===
dt = 1e-2
ts = np.arange(0, 20, dt)

try:
    for t in ts:
        com_target = np.tile(com0, (N_batch, 1))
        # com_target[:, 2] += 0.1 * np.sin(t + np.linspace(0, np.pi, N_batch))
        com_task.target_com = com_target
        # After changes, recompiling the model

        # Solving the instance of the problem
        t0 = perf_counter()
        opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
        t1 = perf_counter()

        # Integrating
        q = integrate_jit(
            mjx_model,
            q,
            opt_solution.v_opt,
            dt,
        )

        # --- MuJoCo visualization ---
        indices = np.arange(0, N_batch, N_batch // vis.n_models)
        vis.update(q[indices])

except KeyboardInterrupt:
    print("Finalizing the simulation as requested...")
except Exception:
    print(traceback.format_exc())
finally:
    if vis.record:
        vis.save_video(round(1 / dt))
    vis.close()
