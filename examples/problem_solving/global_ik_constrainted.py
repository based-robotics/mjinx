import os
from time import perf_counter

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
import optax
from jaxlie import SE3, SO3
from matplotlib import pyplot as plt
from mujoco import viewer

from mjinx import global_ik_step, loss_fn, loss_grad
from mjinx.barriers import ModelJointBarrier, PositionUpperBarrier
from mjinx.tasks import FrameTask

model_path = os.path.abspath(os.path.dirname(__file__)) + "/../robot_descriptions/kuka_iiwa_14/iiwa14.xml"
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
mj_viewer = viewer.launch_passive(
    mj_model,
    mj_data,
    show_left_ui=False,
    show_right_ui=False,
)
q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()
mj_model.jnt_range = [(-20 * np.pi, 20 * np.pi) for _ in range(7)]

renderer.scene.ngeom += 1
mj_viewer.user_scn.ngeom = 1

N_batch = 200
cur_q = jnp.array(
    np.random.uniform(
        q_min,
        q_max,
        size=(mj_model.nq),
    )
)

ee_id = 8

tasks = {
    "ee_task": FrameTask(
        model=mjx_model,
        cost=1 * jnp.eye(6),
        gain=50 * jnp.ones(6),
        frame_id=ee_id,
        target_frame=SE3.from_rotation_and_translation(SO3.identity(), np.array([0.2, 0.2, 0.2])),
    ),
}
barriers = {
    "joint_barrier": ModelJointBarrier(
        model=mjx_model,
        joints_gain=jnp.concat([0.1 * jnp.ones(mj_model.nv), 0.1 * jnp.ones(mj_model.nv)]),
    ),
    "position_barrier": PositionUpperBarrier(
        model=mjx_model,
        gain=jnp.array([0.1]),
        frame_id=ee_id,
        axes="x",
        p_max=np.array([0.3]),
        safe_displacement_gain=1e-2,
    ),
}

opt = optax.adam(learning_rate=0.01)
opt_state = opt.init(q)


print("Compiling")
t0 = perf_counter()
updates, opt_state = global_ik_step(
    model=mjx_model, tasks=tasks, barriers=barriers, q=cur_q, optimizer=opt, state=opt_state, loss_grad=loss_grad
)
print(f"Compilation time: {(perf_counter() - t0) * 1000:.3f}ms")

N_epoch = 5000
try:
    jnts = []
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
            barriers,
            cur_q,
            opt_state,
            loss_grad=loss_grad,
            optimizer=opt,
        )
        print(f"Computation time: {(perf_counter() - t0) * 1000:.3f}ms")
        print(f"Loss: {loss_fn(mjx_model, tasks, barriers, cur_q)}")
        print(f"Position barrier: {mj_data.xpos[ee_id][0]} <= 0.3")
        print("-" * 50)
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

        jnts.append(mj_data.qpos.copy())

except Exception as e:
    print(e.with_traceback(None))
finally:
    mj_viewer.close()

    jnts = np.array(jnts)

    fig, ax = plt.subplots(7, 1, figsize=(10, 20))

    for i in range(7):
        ax[i].axhline(q_min[i], color="r", ls="--", label="Lower limit")
        ax[i].axhline(q_max[i], color="r", ls="--", label="Upper limit")
        ax[i].plot(jnts[:, i], label=f"Joint {i}")
        ax[i].set_title(f"Joint {i}")
        ax[i].set_xlabel("Time (s)")
        ax[i].set_ylabel("Joint position (rad)")
        ax[i].legend()

    plt.tight_layout()
    plt.show()
