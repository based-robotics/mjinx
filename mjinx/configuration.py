#!/usr/bin/env python3

"""..."""

from typing import Sequence
import jax
import jax.numpy as jnp
import jaxlie
import mujoco as mj
from jaxlie import SE3, SO3
from mujoco import mjx

from mjinx.typing import CollisionPair


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


def sorted_pair(x: int, y: int) -> tuple[int, int]:
    return (min(x, y), max(x, y))


def get_distance(model: mjx.Model, data: mjx.Data, collision_pairs: list[CollisionPair]):
    dists = []
    for g1, g2 in collision_pairs:
        types = model.geom_type[g1], model.geom_type[g2]
        data_ids = model.geom_dataid[g1], model.geom_dataid[g2]
        if model.geom_priority[g1] > model.geom_priority[g2]:
            condim = model.geom_condim[g1]
        elif model.geom_priority[g1] < model.geom_priority[g2]:
            condim = model.geom_condim[g2]
        else:
            condim = max(model.geom_condim[g1], model.geom_condim[g2])

        if types[0] == mj.mjtGeom.mjGEOM_HFIELD:
            # add static grid bounds to the grouping key for hfield collisions
            geom_rbound_hfield = model.geom_rbound
            nrow, ncol = model.hfield_nrow[data_ids[0]], model.hfield_ncol[data_ids[0]]
            xsize, ysize = model.hfield_size[data_ids[0]][:2]
            xtick, ytick = (2 * xsize) / (ncol - 1), (2 * ysize) / (nrow - 1)
            xbound = int(jnp.ceil(2 * geom_rbound_hfield[g2] / xtick)) + 1
            xbound = min(xbound, ncol)
            ybound = int(jnp.ceil(2 * geom_rbound_hfield[g2] / ytick)) + 1
            ybound = min(ybound, nrow)
            key = mjx._src.collision_types.FunctionKey(types, data_ids, condim, (xbound, ybound))
        else:
            key = mjx._src.collision_types.FunctionKey(types, data_ids, condim)

        collision_fn = mjx._src.collision_driver._COLLISION_FUNC[sorted_pair(*types)]
        dists.append(collision_fn(model, data, key, jnp.array((g1, g2)).reshape(1, -1))[0])

    return jnp.array(dists).ravel()
