import os

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
import pink
import pinocchio as pin
from jaxlie import SE3
from robot_descriptions.loaders.pinocchio import load_robot_description

from mjinx.configuration import update
from mjinx import tasks

np.printoptions(precision=3, linewidth=1000, suppress=True)

# Load the model
model_path = os.path.abspath(os.path.dirname(__file__)) + "/../robot_descriptions/kuka_iiwa_14/iiwa14.xml"
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

q_init = np.random.uniform(
    mj_model.jnt_range[:, 0],
    mj_model.jnt_range[:, 1],
    size=(mj_model.nq),
)
q_init_mjinx = jnp.array(q_init)

joint_task = tasks.JointTask(
    model=mjx_model,
    cost=jnp.eye(6),
    gain=jnp.ones(mj_model.nv),
    target_q=q,
    mask=jnp.ones(mj_model.nv),
)

reg_task = tasks.RegularizationTask(
    model=mjx_model,
    cost=jnp.eye(6),
    gain=jnp.ones(mj_model.nv),
)


print(joint_task.dim)
print(reg_task.dim)
