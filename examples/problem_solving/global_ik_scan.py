import os
from time import perf_counter

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
import optax
from jaxlie import SE3, SO3

from mjinx import integrate_global_ik, loss_grad
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

cur_q = jnp.zeros(mj_model.nq)

ee_id = 8

dt = 1e-1
N_iters = 20
ts = np.arange(0, N_iters * dt, dt)
print(ts[-1])
tasks = tuple(
    {
        "ee_task": FrameTask(
            model=mjx_model,
            cost=1 * jnp.eye(6),
            gain=1 * jnp.ones(6),
            frame_id=ee_id,
            target_frame=SE3.from_rotation_and_translation(
                SO3.identity(),
                jnp.array(
                    [
                        0.2 + 0.1 * jnp.cos(t),
                        0.2 + 0.1 * jnp.sin(t),
                        0.2,
                    ]
                ),
            ),
        ),
    }
    for t in ts
)
tasks_init = tuple(
    {
        "ee_task": task["ee_task"].copy_and_set(
            target_frame=SE3.from_rotation_and_translation(
                SO3.identity(),
                jnp.array(
                    [
                        0.2 + 0.1,
                        0.2 + 0.1,
                        0.2,
                    ]
                ),
            )
        )
    }
    for task in tasks
)
barriers = tuple({} for _ in ts)

opt = optax.sgd(learning_rate=0.01, momentum=0.9)


print("Compiling inverse kinematics...")
t0 = perf_counter()
res1 = integrate_global_ik(
    mjx_model,
    jnp.array(np.random.uniform(-1, 1, mj_model.nq)),
    tasks_init,
    N_iters_per_task=2,
    optimizer=opt,
    loss_grad=loss_grad,
    dt=dt,
)
print(f"Time taken: {perf_counter() - t0:.3f} seconds")

t0 = perf_counter()
print("Solving inverse kinematics...")
res2 = integrate_global_ik(
    mjx_model,
    cur_q,
    tasks,
    N_iters_per_task=2,
    optimizer=opt,
    loss_grad=loss_grad,
    dt=dt,
)
print(f"Time taken: {perf_counter() - t0:.3f} seconds")

print("Results comparison (if different, value is not cached, and function was compiled):")
print(res1[4, :10])
print(res2[4, :10])
