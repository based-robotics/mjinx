import os

import numpy as np
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx

from mjinx import solve_ik
from mjinx.configuration import update
from mjinx.tasks import ComTask, PositionTask

model_path = os.path.abspath(os.path.dirname(__file__)) + "/robot_descriptions/kuka_iiwa_14/iiwa14.xml"
mj_model = mj.MjModel.from_xml_path(model_path)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.make_data(mjx_model)
mjx_data = update(
    mjx_model,
    jnp.array(
        np.random.uniform(
            mj_model.jnt_range[:, 0],
            mj_model.jnt_range[:, 1],
            size=(mj_model.nq),
        )
    ),
)
N_batch = 10000

tasks = [
    ComTask(cost=jnp.eye(3), gain=jnp.ones(3), lm_damping=jnp.ones(3), target_com=jnp.zeros(3)),
    PositionTask(cost=jnp.eye(3), gain=jnp.ones(3), lm_damping=jnp.ones(3), frame_id=8, target_pos=jnp.zeros(3)),
]

solve_ik(mjx_model, mjx_data, tasks, [], dt=1e-3)
