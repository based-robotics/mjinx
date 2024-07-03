#!/usr/bin/env python3

"""..."""

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx


def update(model: mjx.Model, q: jnp.ndarray) -> mjx.Data:
    """..."""
    data = mjx.make_data(model)
    data.replace(qpos=q)
    data = mjx.kinematics(model, data)
    data = mjx.com_pos(model, data)

    return data


def check_limits(model: mjx.Model, data: mjx.Data) -> jnp.bool:
    """..."""
    return jnp.all(model.jnt_range[:, 0] < data.qpos < model.jnt_range[:, 1])


def get_frame_jacobian(model: mjx.Model, data: mjx.Data, body_id: int) -> jnp.array:
    """Compute pair of (NV, 3) Jacobians of global point attached to body."""

    def fn(carry, b):
        return b if carry is None else b + carry

    mask = (jnp.arange(model.nbody) == body_id) * 1
    # Puts 1 for all parent links of specified body.
    mask = mjx._src.scan.body_tree(model, fn, "b", "b", mask, reverse=True)
    # From all parent links, select only those which add degree of freedoms?..
    mask = mask[jnp.array(model.dof_bodyid)] > 0

    # Subtree_com is the center of mass of the subtree.
    offset = data.xpos[body_id] - data.subtree_com[jnp.array(model.body_rootid)[body_id]]
    # vmap over all degrees of freedom of the subtree.
    jacp = jax.vmap(lambda a, b=offset: a[3:] + jnp.cross(a[:3], b))(data.cdof)
    jacp = jax.vmap(jnp.multiply)(jacp, mask)
    jacr = jax.vmap(jnp.multiply)(data.cdof[:, :3], mask)
    jac = jnp.vstack((jacp, jacr))

    return jac


def get_transform_frame_to_world(model: mjx.Model, data: mjx.Data, frame_id: int) -> jnp.array:
    """..."""
    return jnp.concatenate((data.xpos[frame_id], data.xquat[frame_id, [3, 0, 1, 2]]))


def get_transform(model: mjx.Model, data: mjx.Data, source_id: int, dest_id: int) -> jnp.array:
    """..."""
    t1 = get_transform_frame_to_world(data, source_id)
    t2 = get_transform_frame_to_world(data, dest_id)
    s1, v1 = t1[-1], t1[3:6]
    s2, v2 = t2[-1], t2[3:6]
    delta_q = np.concatenate(
        (
            s1 * s2 + jnp.dot(v1, v2),
            s1 * v2 - s2 * v1 - jnp.cross(v1, v2),
        )
    )
    return jnp.concatenate((t2[:3] - t1[:3], delta_q))


def integrate(model: mjx.Model, data: mjx.Data, velocity: jnp.ndarray, dt: jnp.ndarray) -> jnp.array:
    """..."""
    data = data.replace(qvel=velocity)
    return mjx._src.forward._integrate_pos(model.jnt_type, data.qpos, velocity, dt)


def integrate_inplace(model: mjx.Model, data: mjx.Data, velocity: jnp.ndarray, dt: jnp.ndarray) -> mjx.Data:
    """..."""
    return data.replace(qpos=integrate(model, data, velocity, dt))
