import os
from time import perf_counter

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3

from mjinx import solve_local_ik
from mjinx.tasks import FrameTask

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

N_batch = 200
cur_q = jnp.zeros(mj_model.nq)

ee_id = 8
print(ee_id)

tasks = {
    "ee_task": FrameTask(
        model=mjx_model,
        cost=1 * jnp.eye(6),
        gain=1 * jnp.ones(6),
        frame_id=ee_id,
        target_frame=SE3.from_rotation_and_translation(SO3.identity(), np.array([0.2, 0.2, 0.2])),
    ),
}

dt = 1e-3
ts = np.arange(0, 20, dt)

for t in ts:
    t0 = perf_counter()
    tasks["ee_task"] = tasks["ee_task"].copy_and_set(
        target_frame=SE3.from_rotation_and_translation(
            SO3.identity(),
            np.array(
                [
                    0.2 + 0.1 * jnp.cos(t),
                    0.2 + 0.1 * jnp.sin(t),
                    0.2,
                ]
            ),
        )
    )
    vel = solve_local_ik(mjx_model, cur_q, tasks, {}, damping=1e-12)
    cur_q += vel * dt
    print(f"Time: {(perf_counter() - t0) * 1000 :.3f}ms")
