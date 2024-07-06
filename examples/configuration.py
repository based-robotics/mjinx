import os

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx import configuration

model_path = os.path.abspath(os.path.dirname(__file__)) + "/robot_descriptions/kuka_iiwa_14/iiwa14.xml"
mj_model = mj.MjModel.from_xml_path(model_path)
mjx_model = mjx.put_model(mj_model)


N_batch = 10000
q_rand_np = np.vstack(
    [np.random.uniform(mj_model.jnt_range[:, 0], mj_model.jnt_range[:, 1], size=(mj_model.nq)) for _ in range(N_batch)],
)
v_rand_np = np.vstack(
    [np.random.uniform(-1, 1, size=(mj_model.nv)) for _ in range(N_batch)],
)
q_jnp = jnp.array(q_rand_np)
v_jnp = jnp.array(v_rand_np)


def get_jacobian(model: mjx.Model, q: jnp.ndarray, body_id: int) -> jnp.ndarray:
    return configuration.get_frame_jacobian(model, configuration.update(model, q), body_id)


update_batched = jax.jit(jax.vmap(configuration.update, in_axes=(None, 0)))
get_jacobian_batched = jax.jit(jax.vmap(get_jacobian, in_axes=(None, 0, None)))

jac_batched = get_jacobian_batched(mjx_model, q_jnp, 8)

print(jac_batched[0])
