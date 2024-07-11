import os

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from time import perf_counter

from mjinx import solve_ik
from mjinx.tasks import ComTask, PositionTask
from robot_descriptions.loaders.pinocchio import load_robot_description

import pink
import qpsolvers
import pinocchio as pin

# Load the model
model_path = os.path.abspath(os.path.dirname(__file__)) + "/robot_descriptions/kuka_iiwa_14/iiwa14.xml"
mj_model = mj.MjModel.from_xml_path(model_path)
mjx_model = mjx.put_model(mj_model)
q = jnp.array(
    np.random.uniform(
        mj_model.jnt_range[:, 0],
        mj_model.jnt_range[:, 1],
        size=(mj_model.nq),
    )
)
pin_robot = load_robot_description("iiwa14_description")

N_batch = 1000
q_batched = np.vstack(
    [
        jnp.array(
            np.random.uniform(
                mj_model.jnt_range[:, 0],
                mj_model.jnt_range[:, 1],
                size=(mj_model.nq),
            )
        )
        for _ in range(N_batch)
    ]
)


tasks = (ComTask(cost=jnp.eye(3), gain=jnp.ones(3), lm_damping=jnp.ones(3), target_com=jnp.zeros(3)),)

solve_ik_batched = jax.jit(
    jax.vmap(
        solve_ik,
        in_axes=(
            None,
            0,
            None,
            None,
            None,
            None,
        ),
    ),
    static_argnums=[4, 5],
)

solution = solve_ik_batched(mjx_model, jnp.array(q_batched), tasks, (), 1e-3, 1e-12)
print(solution[0])

configuration = pink.Configuration(pin_robot.model, pin_robot.data, q_batched[0])

pink_com_task = pink.tasks.ComTask(
    cost=1.0,
    gain=np.ones(3),
    lm_damping=1.0,
)
pink_com_task.set_target(np.zeros(3))

solver = qpsolvers.available_solvers[0]
if "osqp" in qpsolvers.available_solvers:
    solver = "osqp"
velocity = pink.solve_ik(
    configuration,
    [pink_com_task],
    dt=1e-3,
    solver=solver,
    damping=1e-12,
)

print(velocity)
