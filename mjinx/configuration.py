#!/usr/bin/env python3

"""..."""

import jax
import jax.numpy as jnp
import jaxlie
import mujoco as mj
from jaxlie import SE3, SO3
from mujoco import mjx


def update(model: mjx.Model, q: jnp.ndarray) -> mjx.Data:
    """..."""
    data = mjx.make_data(model)
    data = data.replace(qpos=q)
    data = mjx.fwd_position(model, data)
    data = mjx.com_pos(model, data)

    return data


# TODO: not working
# def check_limits(model: mjx.Model, data: mjx.Data) -> bool:
#     """..."""
#     return (model.jnt_range[:, 0] < data.qpos < model.jnt_range[:, 1]).all().item()


def get_frame_jacobian_world_aligned(model: mjx.Model, data: mjx.Data, body_id: int) -> jnp.ndarray:
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


def get_frame_jacobian_local(model: mjx.Model, data: mjx.Data, body_id: int) -> jax.Array:
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
    return get_transform_frame_to_world(model, data, dest_id) @ get_transform_frame_to_world(model, data, source_id)


def integrate(model: mjx.Model, q0: jnp.ndarray, velocity: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    """..."""
    return mjx._src.forward._integrate_pos(model.jnt_type, q0, velocity, dt)


def get_configuration_limit(model: mjx.Model, limit: jnp.ndarray | float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """..."""
    # -limit <= v <- limit
    limit_array = jnp.ones(model.nv) * limit if isinstance(limit, float) else limit

    return (
        jnp.vstack((-1 * jnp.eye(model.nv), jnp.eye(model.nv))),
        jnp.concatenate((limit_array, limit_array)),
    )


def get_joint_zero(model: mjx.Model) -> jnp.ndarray:
    """..."""
    jnts = []

    for jnt_id in range(model.njnt):
        jnt_type = model.jnt_type[jnt_id]
        match jnt_type:
            case mj.mjtJoint.mjJNT_FREE:
                jnts.append(jnp.array([0, 0, 0, 1, 0, 0, 0]))
            case mj.mjtJoint.mjJNT_BALL:
                jnts.append(jnp.array([1, 0, 0, 0]))
            case mj.mjtJoint.mjJNT_HINGE | mj.mjtJoint.mjJNT_SLIDE:
                jnts.append(jnp.zeros(1))

    return jnp.concatenate(jnts)


def joint_difference(model: mjx.Model, q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    jnt_diff = []
    idx = 0
    for jnt_id in range(model.njnt):
        jnt_type = model.jnt_type[jnt_id]
        match jnt_type:
            case mj.mjtJoint.mjJNT_FREE:
                q1_pos, q1_quat = q1[idx : idx + 3], q1[idx + 3 : idx + 7]
                q2_pos, q2_quat = q2[idx : idx + 3], q2[idx + 3 : idx + 7]
                indices = jnp.array([1, 2, 3, 0])

                frame1_SE3: SE3 = SE3.from_rotation_and_translation(
                    SO3.from_quaternion_xyzw(q1_quat[indices]),
                    q1_pos,
                )
                frame2_SE3: SE3 = SE3.from_rotation_and_translation(
                    SO3.from_quaternion_xyzw(q2_quat[indices]),
                    q2_pos,
                )

                jnt_diff.append(jaxlie.manifold.rminus(frame1_SE3, frame2_SE3))
                idx += 7
            case mj.mjtJoint.mjJNT_BALL:
                q1_quat = q1[idx : idx + 4]
                q2_quat = q2[idx : idx + 4]
                indices = jnp.array([1, 2, 3, 0])

                frame1_SO3: SO3 = SO3.from_quaternion_xyzw(q1_quat[indices])
                frame2_SO3: SO3 = SO3.from_quaternion_xyzw(q2_quat[indices])

                jnt_diff.append(jaxlie.manifold.rminus(frame1_SO3, frame2_SO3))
                idx += 4
            case mj.mjtJoint.mjJNT_HINGE | mj.mjtJoint.mjJNT_SLIDE:
                jnt_diff.append(q1[idx : idx + 1] - q2[idx : idx + 1])
                idx += 1

    return jnp.concatenate(jnt_diff)
