import mujoco as mj
import mujoco.mjx as mjx
from robot_descriptions import cassie_mj_description

from mjinx import configuration

model_path = cassie_mj_description.MJCF_PATH
mj_model = mj.MjModel.from_xml_path(model_path)
mjx_model = mjx.put_model(mj_model)

q = configuration.get_joint_zero(mj_model)
print(q)
q1 = q.copy()

diff = configuration.joint_difference(mjx_model, q, q1)
print(q)
print(diff)
