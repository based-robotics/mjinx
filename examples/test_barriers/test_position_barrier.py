import os

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
import pink
import pink.barriers
import pinocchio as pin
from jaxlie import SE3
from robot_descriptions.loaders.pinocchio import load_robot_description

from mjinx.barriers import PositionUpperBarrier
from mjinx.configuration import update

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

configuration = pink.Configuration(pin_robot.model, pin_robot.data, q_init)
mjx_data = update(mjx_model, q_init_mjinx)

frame_task_pink = pink.barriers.PositionBarrier(
    "iiwa_link_7",
    p_max=0.1 * np.ones(3),
    gain=1,
)
frame_task_mjinx = PositionUpperBarrier(
    model=mjx_model,
    frame_id=8,
    gain=jnp.ones(6),
    p_max=0.1 * jnp.ones(3),
)

print("Task error:")
print("Pink:")
print(frame_task_pink.compute_barrier(configuration))
print("Mjinx:")
print(frame_task_mjinx.compute_barrier(mjx_data))
print("Task Jacobian:")

jac_pink = frame_task_pink.compute_jacobian(configuration)
jac_mjinx = frame_task_mjinx.compute_jacobian(mjx_data)
print("=== Jacobian ===")
print("Pink:")
print(jac_pink)
print("Mjinx:")
print(jac_mjinx)
