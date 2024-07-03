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

update_jit = jax.jit(jax.vmap(configuration.update, in_axes=(None, 0)))
get_frame_jacobian_jit = jax.jit(
    jax.vmap(configuration.get_transform_frame_to_world, in_axes=(None, 0, None)), static_argnums=(2,)
)
integrate_inplace_jit = jax.jit(
    jax.vmap(configuration.integrate_inplace, in_axes=(None, 0, 0, None)), static_argnums=(3,)
)


mj_data_batch = update_jit(mjx_model, q_rand_np)
mj_jacobian_batch = get_frame_jacobian_jit(mjx_model, mj_data_batch, 2)
mj_data_batch = integrate_inplace_jit(mjx_model, mj_data_batch, v_jnp, 1e-3)
