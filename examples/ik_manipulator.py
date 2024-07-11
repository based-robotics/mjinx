import os
from time import perf_counter

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3
from mujoco import viewer

from mjinx import solve_ik
from mjinx.tasks import ComTask, FrameTask, PositionTask
from mjinx import configuration

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

mj_data = mj.MjData(mj_model)
renderer = mj.Renderer(mj_model)
mj_viewer = mj.viewer.launch_passive(
    mj_model,
    mj_data,
    show_left_ui=False,
    show_right_ui=False,
)
renderer.scene.ngeom += 1
mj_viewer.user_scn.ngeom = 1

N_batch = 200
cur_q = jnp.zeros(mj_model.nq)

ee_id = 8
print(ee_id)

tasks = (
    FrameTask(
        cost=1 * jnp.eye(6),
        gain=1 * jnp.ones(6),
        frame_id=ee_id,
        target_frame=SE3.from_rotation_and_translation(SO3.identity(), np.array([0.2, 0.2, 0.2])),
    ),
)

ts = np.arange(0, 20, 1e-3)

mj.mjv_initGeom(
    mj_viewer.user_scn.geoms[0],
    mj.mjtGeom.mjGEOM_SPHERE,
    0.1 * np.ones(3),
    np.array([0.2, 0.2, 0.2]),
    np.eye(3).flatten(),
    np.array([0.565, 0.933, 0.565, 0.4]),
)
# solve_ik_jitted = jax.jit(solve_ik, static_argnums=[4, 5])

try:
    # Warm-up JIT
    for _ in ts:
        vel = solve_ik(mjx_model, cur_q, tasks, (), damping=1e-12)
        cur_q += vel * 5 * 1e-1
        mj_data.qpos = cur_q
        mj_data.qvel = vel

        # Run the forward dynamics to reflec
        # the updated state in the data
        mj.mj_forward(mj_model, mj_data)
        print(cur_q)
        mj_viewer.sync()
except Exception as e:
    print(e)
    mj_viewer.close()
