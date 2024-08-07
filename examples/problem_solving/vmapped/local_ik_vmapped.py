import os
from time import perf_counter

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3
from mujoco import viewer

from mjinx import solve_local_ik
from mjinx.tasks import FrameTask

model_path = os.path.abspath(os.path.dirname(__file__)) + "/../../robot_descriptions/kuka_iiwa_14/iiwa14.xml"
mj_model = mj.MjModel.from_xml_path(model_path)
mjx_model = mjx.put_model(mj_model)

N_batch = 2000
cur_q = jnp.array(
    [
        np.random.uniform(
            mj_model.jnt_range[:, 0],
            mj_model.jnt_range[:, 1],
            size=(mj_model.nq),
        )
        for _ in range(N_batch)
    ]
)
print(cur_q.shape)
mj_data = mj.MjData(mj_model)
renderer = mj.Renderer(mj_model)
mj_viewer = viewer.launch_passive(
    mj_model,
    mj_data,
    show_left_ui=False,
    show_right_ui=False,
)
renderer.scene.ngeom += 1
mj_viewer.user_scn.ngeom = 1

ee_id = 8

tasks = {
    "ee_task": FrameTask(
        model=mjx_model,
        cost=1 * jnp.eye(6),
        gain=1 * jnp.ones(6),
        frame_id=ee_id,
        target_frame=SE3.from_rotation_and_translation(SO3.identity(), np.array([0.2, 0.2, 0.2])),
    ),
}

dt = 1e-2
ts = np.arange(0, 20, dt)

mj.mjv_initGeom(
    mj_viewer.user_scn.geoms[0],
    mj.mjtGeom.mjGEOM_SPHERE,
    0.1 * np.ones(3),
    np.array([0.2, 0.2, 0.2]),
    np.eye(3).flatten(),
    np.array([0.565, 0.933, 0.565, 0.4]),
)

solve_local_ik_vmap = jax.jit(
    jax.vmap(solve_local_ik, in_axes=(None, 0, None, None, None, None)), static_argnames=("damping", "solver")
)

# try:
# Warm-up JIT
for t in ts:
    tasks["ee_task"] = tasks["ee_task"].copy_and_set(
        target_frame=SE3.from_rotation_and_translation(
            SO3.identity(),
            np.array(
                [
                    0.3 + 0.1 * jnp.cos(t),
                    0.3 + 0.1 * jnp.sin(t),
                    0.2,
                ]
            ),
        )
    )
    t0 = perf_counter()
    vel = solve_local_ik_vmap(mjx_model, cur_q, tasks, {}, 1e-12, "osqp")
    print(f"Computation time: {(perf_counter() - t0) * 1000:.3f}ms")
    print(f"Number of diverged problems: {jnp.isnan(vel).any(axis=1).sum()}")
    cur_q += vel * dt
    mj_data.qpos = cur_q[0]
    mj_data.qvel = vel[0]
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
# except Exception as e:
#     print(e.with_traceback(None))
#     mj_viewer.close()
