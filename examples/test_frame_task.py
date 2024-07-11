import os
from time import perf_counter

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
import pink
import pinocchio as pin
import qpsolvers
from jaxlie import SE3, SO3
from robot_descriptions.loaders.pinocchio import load_robot_description

from mjinx import solve_ik
from mjinx.configuration import update
from mjinx.tasks import FrameTask

np.printoptions(precision=3, linewidth=1000, suppress=True)

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

q_init = np.random.uniform(
    mj_model.jnt_range[:, 0],
    mj_model.jnt_range[:, 1],
    size=(mj_model.nq),
)
q_init_mjinx = jnp.array(q_init)

configuration = pink.Configuration(pin_robot.model, pin_robot.data, q_init)
mjx_data = update(mjx_model, q_init_mjinx)

frame_task_pink = pink.FrameTask(
    frame="iiwa_link_7", position_cost=np.ones(3), orientation_cost=np.ones(3), gain=np.eye(3)
)
frame_task_pink.set_target(pin.XYZQUATToSE3([0, 0, 0, 0, 0, 0, 1]))
frame_task_mjinx = FrameTask(
    frame_id=8,
    gain=jnp.ones(6),
    cost=jnp.eye(6),
    lm_damping=jnp.ones(6),
    target_frame=SE3.identity(),
)

print("Task error:")
print("Pink:")
print(frame_task_pink.compute_error(configuration))
print("Mjinx:")
print(frame_task_mjinx.compute_error(mjx_model, mjx_data))
print("Task Jacobian:")

jlog6_pink, frame_jac_pink = frame_task_pink.compute_jacobian(configuration)
jlog6_mjinx, frame_jac_mjinx = frame_task_mjinx.compute_jacobian(mj_model, mjx_data)
print("=== Frame jac ===")
print("Pink:")
print(frame_jac_pink)
print("Mjinx:")
print(frame_jac_mjinx)
print("=== Jacobian log ===")
print("Pink:")
print(jlog6_pink)
print("Mjinx:")
print(jlog6_mjinx)
