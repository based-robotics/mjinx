#!/usr/bin/env python3

"""..."""

import jax
import jax.numpy as jnp
from jaxlie import SE3, SO3
from mujoco import mjx


def update(model: mjx.Model, q: jnp.ndarray) -> mjx.Data:
    """..."""
    data = mjx.make_data(model)
    data = data.replace(qpos=q)
    data = mjx.fwd_position(model, data)
    data = mjx.com_pos(model, data)

    return data


def check_limits(model: mjx.Model, data: mjx.Data) -> jnp.bool:
    """..."""
    return jnp.all(model.jnt_range[:, 0] < data.qpos < model.jnt_range[:, 1])


def get_frame_jacobian_world_aligned(model: mjx.Model, data: mjx.Data, body_id: int) -> jnp.array:
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

    return jnp.vstack((jacp.T, jacr.T)).T


def get_frame_jacobian_local(model: mjx.Model, data: mjx.Data, body_id: jax.Array) -> tuple[jax.Array, jax.Array]:
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

    # Get rotation matrix, which describes rotation of local frame
    R_inv = data.xmat[body_id].reshape(3, 3).T

    # vmap over all degrees of freedom of the subtree.
    jacp = jax.vmap(lambda a, b=offset, R=R_inv: R @ (a[3:] + jnp.cross(a[:3], b)))(data.cdof)
    jacp = jax.vmap(jnp.multiply)(jacp, mask)
    jacr = jax.vmap(jnp.multiply)(data.cdof[:, :3] @ R_inv.T, mask)

    return jnp.vstack((jacp.T, jacr.T)).T


def get_transform_frame_to_world(model: mjx.Model, data: mjx.Data, frame_id: int) -> SE3:
    """..."""
    return SE3.from_rotation_and_translation(
        SO3.from_quaternion_xyzw(data.xquat[frame_id, [1, 2, 3, 0]]),
        data.xpos[frame_id],
    )


def get_transform(model: mjx.Model, data: mjx.Data, source_id: int, dest_id: int) -> SE3:
    """..."""
    return get_transform_frame_to_world(data, dest_id) @ get_transform_frame_to_world(data, source_id)


def integrate(model: mjx.Model, data: mjx.Data, velocity: jnp.ndarray, dt: jnp.ndarray) -> jnp.array:
    """..."""
    data = data.replace(qvel=velocity)
    return mjx._src.forward._integrate_pos(model.jnt_type, data.qpos, velocity, dt)


def integrate_inplace(model: mjx.Model, data: mjx.Data, velocity: jnp.ndarray, dt: jnp.ndarray) -> mjx.Data:
    """..."""
    return data.replace(qpos=integrate(model, data, velocity, dt))


def get_configuration_limit(model: mjx.Model, limit: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """..."""
    # -limit <= v <- limit

    return (
        jnp.vstack(
            (
                -1 * jnp.eye(model.nv),
                jnp.eye(model.nv),
            )
        ),
        jnp.hstack(
            (
                limit,
                limit,
            )
        ),
    )
