import os

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3
from time import perf_counter

from mjinx import solve_ik
from mjinx.tasks import ComTask, PositionTask, FrameTask

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

N_batch = 200
q_batched = jnp.vstack(
    [
        jnp.array(
            np.random.uniform(
                mj_model.jnt_range[:, 0],
                mj_model.jnt_range[:, 1],
                size=(mj_model.nq),
            )
        )
        for _ in range(N_batch)
    ]
)

tasks = (
    ComTask(cost=jnp.eye(3), gain=jnp.ones(3), lm_damping=jnp.ones(3), target_com=jnp.zeros(3)),
    PositionTask(cost=jnp.eye(3), gain=jnp.ones(3), lm_damping=jnp.ones(3), frame_id=8, target_pos=jnp.zeros(3)),
    FrameTask(cost=jnp.eye(6), gain=jnp.ones(6), lm_damping=jnp.ones(6), frame_id=8, target_frame=SE3.identity()),
)

solve_ik_batched = jax.jit(
    jax.vmap(
        solve_ik,
        in_axes=(
            None,
            0,
            None,
            None,
            None,
            None,
        ),
    ),
    static_argnums=[4, 5],
)

t0 = perf_counter()
# Warm-up JIT
solve_ik_batched(mjx_model, q_batched, tasks, (), 1e-3, 1e-12)
print((perf_counter() - t0) * 1000)
print("-" * 20)

for _ in range(5):
    q_batched = jnp.vstack(
        [
            jnp.array(
                np.random.uniform(
                    mj_model.jnt_range[:, 0],
                    mj_model.jnt_range[:, 1],
                    size=(mj_model.nq),
                )
            )
            for _ in range(N_batch)
        ]
    )
    t0 = perf_counter()
    solve_ik_batched(mjx_model, q_batched, tasks, (), 1e-3, 1e-12)
    print((perf_counter() - t0) * 1000)
