import os

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx import barriers
from mjinx.configuration import update

np.printoptions(precision=3, linewidth=1000, suppress=True)

# Load the model
model_path = os.path.abspath(os.path.dirname(__file__)) + "/../robot_descriptions/kuka_iiwa_14/iiwa14.xml"
mj_model = mj.MjModel.from_xml_path(model_path)
mjx_model = mjx.put_model(mj_model)

q_init_mjinx = jnp.zeros(mj_model.nq)

joint_barrier = barriers.ModelJointBarrier(
    model=mjx_model,
    joints_gain=jnp.ones(mj_model.nv),
    safe_displacement_gain=0.0,
)

print(joint_barrier.qmin)
print(joint_barrier.qmax)

print(joint_barrier.compute_barrier(update(mjx_model, q_init_mjinx)))
print(joint_barrier.compute_jacobian(update(mjx_model, q_init_mjinx)))
