import traceback

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3

from mjinx.components.barriers import JointBarrier
from mjinx.components.tasks import FrameTask
from mjinx.configuration import integrate, update
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver
from mjinx.visualize import BatchVisualizer

# === Mujoco ===

mj_model = mj.MjModel.from_xml_path("examples/g1_description/g1.xml")
mjx_model = mjx.put_model(mj_model)
print(mjx_model.nq, mjx_model.nv)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

# --- Mujoco visualization ---
# Initialize render window and launch it at the background
vis = BatchVisualizer("examples/g1_description/g1.xml", n_models=5, alpha=0.2)

# === Mjinx ===
# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-1, v_max=1)

# Creating components of interest and adding them to the problem
body_task = FrameTask("body_task", cost=1.0, gain=10, obj_name="torso_link")
joints_barrier = JointBarrier("jnt_range", gain=0.1, floating_base=True)

# Arms (moving)
left_arm_task = FrameTask(
    "left_arm_task",
    cost=1.0,
    gain=10,
    obj_type=mj.mjtObj.mjOBJ_SITE,
    obj_name="left_wrist",
)
right_arm_task = FrameTask(
    "right_arm_task",
    cost=1.0,
    gain=10,
    obj_type=mj.mjtObj.mjOBJ_SITE,
    obj_name="right_wrist",
)

# Feet (in stance)
left_foot_task = FrameTask(
    "left_foot_task",
    cost=20.0,
    gain=10.0,
    obj_type=mj.mjtObj.mjOBJ_SITE,
    obj_name="left_foot",
)
right_foot_task = FrameTask(
    "right_foot_task",
    cost=20.0,
    gain=10.0,
    obj_type=mj.mjtObj.mjOBJ_SITE,
    obj_name="right_foot",
)


problem.add_component(body_task)
problem.add_component(joints_barrier)
problem.add_component(left_arm_task)
problem.add_component(right_arm_task)
problem.add_component(left_foot_task)
problem.add_component(right_foot_task)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = LocalIKSolver(mjx_model, maxiter=10)

# Initializing initial condition
N_batch = 5
q0 = mj_model.keyframe("stand").qpos
q = jnp.tile(q0, (N_batch, 1))

# TODO: implement update_from_model_data
mjx_data = update(mjx_model, jnp.array(q0))

left_foot_pos = mjx_data.geom_xpos[mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_SITE, "left_foot")]
problem.component("left_foot_task").target_frame = jnp.array([*left_foot_pos, 1, 0, 0, 0])

right_foot_pos = mjx_data.geom_xpos[mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_SITE, "right_foot")]
problem.component("right_foot_task").target_frame = jnp.array([*right_foot_pos, 1, 0, 0, 0])

left_wrist_id = mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_SITE, "left_wrist")
left_wrist_0 = SE3.from_rotation_and_translation(
    rotation=SO3.from_matrix(mjx_data.site_xmat[left_wrist_id]),
    translation=mjx_data.site_xpos[left_wrist_id],
)
right_wrist_id = mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_SITE, "right_wrist")
right_wrist_0 = SE3.from_rotation_and_translation(
    rotation=SO3.from_matrix(mjx_data.site_xmat[right_wrist_id]),
    translation=mjx_data.site_xpos[right_wrist_id],
)

# --- Batching ---
solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["left_arm_task"].target_frame = 0
    empty_problem_data.components["right_arm_task"].target_frame = 0

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
        print(left_wrist_0.wxyz_xyz.shape)
        # Changing desired values
        left_arm_task.target_frame = jnp.tile(
            left_wrist_0.wxyz_xyz + jnp.array([0, 0, 0, 0, 0, 0.2 * jnp.sin(t), 0]), (N_batch, 1)
        )
        right_arm_task.target_frame = jnp.tile(
            right_wrist_0.wxyz_xyz + jnp.array([0, 0, 0, 0, 0, -0.2 * jnp.sin(t), 0]), (N_batch, 1)
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
