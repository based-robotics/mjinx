import os
from functools import partial
from time import perf_counter
from typing import Callable

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
import optax
from jaxlie import SE3, SO3
from mujoco import viewer

from mjinx import configuration, global_ik_step, loss_grad
from mjinx.tasks import FrameTask, Task

model_path = os.path.abspath(os.path.dirname(__file__)) + "/robot_descriptions/kuka_iiwa_14/iiwa14.xml"
mj_model = mj.MjModel.from_xml_path(model_path)
mjx_model = mjx.put_model(mj_model)
q = jnp.array(
    np.random.uniform(
        -1,
        1,
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
cur_q = jnp.array(
    np.random.uniform(
        mj_model.jnt_range[:, 0],
        mj_model.jnt_range[:, 1],
        size=(mj_model.nq),
    )
)

ee_id = 8
print(ee_id)

tasks = {
    "ee_task": FrameTask(
        cost=1 * jnp.eye(6),
        gain=1 * jnp.ones(6),
        frame_id=ee_id,
        target_frame=SE3.from_rotation_and_translation(SO3.identity(), np.array([0.2, 0.2, 0.2])),
    ),
}

opt = optax.sgd(learning_rate=0.01, momentum=0.9)
opt_state = opt.init(q)


print("Compiling")
t0 = perf_counter()
updates, opt_state = global_ik_step(
    model=mjx_model, tasks=tasks, q=cur_q, optimizer=opt, state=opt_state, loss_grad=loss_grad
)
print(f"Compilation time: {(perf_counter() - t0) * 1000:.3f}ms")

N_epoch = 5000
try:
    for i in range(N_epoch):
        if i % 5 == 0:
            tasks["ee_task"] = tasks["ee_task"].copy_and_set(
                target_frame=SE3.from_rotation_and_translation(
                    SO3.identity(),
                    np.array(
                        [
                            0.3 + 0.1 * jnp.cos(i / 100),
                            0.3 + 0.1 * jnp.sin(i / 100),
                            0.2,
                        ]
                    ),
                )
            )

        t0 = perf_counter()
        cur_q, opt_state = global_ik_step(
            mjx_model,
            tasks,
            cur_q,
            opt_state,
            loss_grad=loss_grad,
            optimizer=opt,
        )
        print(f"Computation time: {(perf_counter() - t0) * 1000:.3f}ms")
        print(cur_q)
        mj_data.qpos = cur_q
        mj.mjv_initGeom(
            mj_viewer.user_scn.geoms[0],
            mj.mjtGeom.mjGEOM_SPHERE,
            0.1 * np.ones(3),
            np.array(tasks["ee_task"].target_frame.translation(), dtype=np.float64),
            np.eye(3).flatten(),
            np.array([0.565, 0.933, 0.565, 0.4]),
        )

        # Run the forward dynamics to reflec
        # the updated state in the data
        mj.mj_forward(mj_model, mj_data)
        mj_viewer.sync()
except Exception as e:
    print(e.with_traceback(None))
    mj_viewer.close()
