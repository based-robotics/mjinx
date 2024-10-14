"""Helper functions to work with MuJoCo models."""

import jax
import jax.numpy as jnp
import jaxlie
import mujoco as mj
from jaxlie import SE3, SO3
from mujoco import mjx

from mjinx.typing import CollisionPair


def update(model: mjx.Model, q: jnp.ndarray) -> mjx.Data:
    """
    Update the MuJoCo data with new joint positions.

    :param model: The MuJoCo model.
    :param q: The new joint positions.
    :return: Updated MuJoCo data.
    """
    data = mjx.make_data(model)
    data = data.replace(qpos=q)
    data = mjx.fwd_position(model, data)
    data = mjx.com_pos(model, data)

    return data


def get_frame_jacobian_world_aligned(
    model: mjx.Model, data: mjx.Data, obj_id: int, obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY
) -> jnp.ndarray:
    """
    Compute pair of (NV, 3) Jacobians of global point attached to body.

    :param model: The MuJoCo model.
    :param data: The MuJoCo data.
    :param body_id: The ID of the body.
    :return: The Jacobian matrix.
    """

    def fn(carry, b):
        return b if carry is None else b + carry

    body_id: int
    obj_des_pos: jnp.ndarray
    match obj_type:
        case mj.mjtObj.mjOBJ_GEOM:
            body_id = model.geom_bodyid[obj_id]
            obj_des_pos = data.geom_xpos[obj_id]
        case mj.mjtObj.mjOBJ_SITE:
            body_id = model.site_bodyid[obj_id]
            obj_des_pos = data.site_xpos[obj_id]
        case _:  # default -- mjOBJ_BODY:
            body_id = obj_id
            obj_des_pos = data.xpos[obj_id]

    mask = (jnp.arange(model.nbody) == body_id) * 1
    # Puts 1 for all parent links of specified body.
    mask = mjx._src.scan.body_tree(model, fn, "b", "b", mask, reverse=True)
    # From all parent links, select only those which add degree of freedoms?..
    mask = mask[jnp.array(model.dof_bodyid)] > 0

    # Subtree_com is the center of mass of the subtree.
    offset = obj_des_pos - data.subtree_com[jnp.array(model.body_rootid)[body_id]]
    # vmap over all degrees of freedom of the subtree.
    jacp = jax.vmap(lambda a, b=offset: a[3:] + jnp.cross(a[:3], b))(data.cdof)
    jacp = jax.vmap(jnp.multiply)(jacp, mask)
    jacr = jax.vmap(jnp.multiply)(data.cdof[:, :3], mask)

    return jnp.vstack((jacp.T, jacr.T)).T


def get_frame_jacobian_local(model: mjx.Model, data: mjx.Data, obj_id: int, obj_type: mj.mjtObj) -> jax.Array:
    """
    Compute pair of (NV, 3) Jacobians of global point attached to body in local frame.

    :param model: The MuJoCo model.
    :param data: The MuJoCo data.
    :param body_id: The ID of the body.
    :return: The Jacobian matrix in local frame.
    """

    def fn(carry, b):
        return b if carry is None else b + carry

    body_id: int
    obj_des_pos: jnp.ndarray
    obj_des_rot: jnp.ndarray
    match obj_type:
        case mj.mjtObj.mjOBJ_GEOM:
            body_id = model.geom_bodyid[obj_id]
            obj_des_pos = data.geom_xpos[obj_id]
            obj_des_rot = data.geom_xmat[obj_id]
        case mj.mjtObj.mjOBJ_SITE:
            body_id = model.site_bodyid[obj_id]
            obj_des_pos = data.site_xpos[obj_id]
            obj_des_rot = data.site_xmat[obj_id]
        case _:  # default -- mjOBJ_BODY:
            body_id = obj_id
            obj_des_pos = data.xpos[obj_id]
            obj_des_rot = data.xmat[obj_id]

    mask = (jnp.arange(model.nbody) == body_id) * 1
    # Puts 1 for all parent links of specified body.
    mask = mjx._src.scan.body_tree(model, fn, "b", "b", mask, reverse=True)
    # From all parent links, select only those which add degree of freedoms?..
    mask = mask[jnp.array(model.dof_bodyid)] > 0

    # Subtree_com is the center of mass of the subtree.
    offset = obj_des_pos - data.subtree_com[jnp.array(model.body_rootid)[body_id]]

    # Get rotation matrix, which describes rotation of local frame
    R_inv = obj_des_rot.reshape(3, 3).T

    # vmap over all degrees of freedom of the subtree.
    jacp = jax.vmap(lambda a, b=offset, R=R_inv: R @ (a[3:] + jnp.cross(a[:3], b)))(data.cdof)
    jacp = jax.vmap(jnp.multiply)(jacp, mask)
    jacr = jax.vmap(jnp.multiply)(data.cdof[:, :3] @ R_inv.T, mask)

    return jnp.vstack((jacp.T, jacr.T)).T


def get_transform_frame_to_world(model: mjx.Model, data: mjx.Data, frame_id: int) -> SE3:
    """
    Get the transformation from frame to world coordinates.

    :param model: The MuJoCo model.
    :param data: The MuJoCo data.
    :param frame_id: The ID of the frame.
    :return: The SE3 transformation.
    """
    return SE3.from_rotation_and_translation(
        SO3.from_quaternion_xyzw(data.xquat[frame_id, [1, 2, 3, 0]]),
        data.xpos[frame_id],
    )


def get_transform(model: mjx.Model, data: mjx.Data, source_id: int, dest_id: int) -> SE3:
    """
    Get the transformation between two frames.

    :param model: The MuJoCo model.
    :param data: The MuJoCo data.
    :param source_id: The ID of the source frame.
    :param dest_id: The ID of the destination frame.
    :return: The SE3 transformation from source to destination.
    """
    return get_transform_frame_to_world(model, data, dest_id) @ get_transform_frame_to_world(model, data, source_id)


def integrate(model: mjx.Model, q0: jnp.ndarray, velocity: jnp.ndarray, dt: jnp.ndarray | float) -> jnp.ndarray:
    """
    Integrate the joint positions given initial position, velocity, and time step.

    :param model: The MuJoCo model.
    :param q0: The initial joint positions.
    :param velocity: The joint velocities.
    :param dt: The time step.
    :return: The integrated joint positions.
    """
    return mjx._src.forward._integrate_pos(model.jnt_type, q0, velocity, dt)


def get_configuration_limit(model: mjx.Model, limit: jnp.ndarray | float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get the configuration limits for the model.

    :param model: The MuJoCo model.
    :param limit: The limit value(s).
    :return: A tuple of arrays representing the lower and upper bounds.
    """
    # -limit <= v <- limit
    limit_array = jnp.ones(model.nv) * limit if isinstance(limit, float) else limit

    return (
        jnp.vstack((-1 * jnp.eye(model.nv), jnp.eye(model.nv))),
        jnp.concatenate((limit_array, limit_array)),
    )


def get_joint_zero(model: mjx.Model) -> jnp.ndarray:
    """
    Get the zero configuration for all joints in the model.

    :param model: The MuJoCo model.
    :return: An array representing the zero configuration for all joints.
    """
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
    """
    Compute the difference between two joint configurations.

    :param model: The MuJoCo model.
    :param q1: The first joint configuration.
    :param q2: The second joint configuration.
    :return: The difference between the two configurations.
    """
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
    """
    Return a sorted pair of integers.

    :param x: The first integer.
    :param y: The second integer.
    :return: A tuple of the two integers, sorted in ascending order.
    """
    return (min(x, y), max(x, y))


def get_distance(model: mjx.Model, data: mjx.Data, collision_pairs: list[CollisionPair]) -> jnp.ndarray:
    """
    Compute the distances for the given collision pairs.

    :param model: The MuJoCo model.
    :param data: The MuJoCo data.
    :param collision_pairs: A list of collision pairs to check.
    :return: An array of distances for each collision pair.
    """
    dists = []
    for g1, g2 in collision_pairs:
        if model.geom_type[g1] > model.geom_type[g2]:
            g1, g2 = g2, g1
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
            raise NotImplementedError("Height field is not yet supported for collision detection")
        key = mjx._src.collision_types.FunctionKey(types, data_ids, condim)

        collision_fn = mjx._src.collision_driver._COLLISION_FUNC[types]
        dists.append(
            collision_fn(
                model,
                data,
                key,
                jnp.array((g1, g2)).reshape(1, -1),
            )[0].min()
        )
    return jnp.array(dists).ravel()


def skew_symmetric(v: jnp.ndarray) -> jnp.ndarray:
    """
    Create a skew-symmetric matrix from a 3D vector.

    This function takes a 3D vector and returns its corresponding 3x3 skew-symmetric matrix.
    The skew-symmetric matrix is used in various robotics and physics calculations,
    particularly for cross products and rotations.

    :param v: A 3D vector (3x1 array).
    :return: A 3x3 skew-symmetric matrix.
    """

    return jnp.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def attitude_jacobian(q: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the attitude Jacobian for a quaternion.

    This function calculates the 4x3 attitude Jacobian matrix for a given unit quaternion.
    The attitude Jacobian is used in robotics and computer vision for relating
    changes in orientation (represented by quaternions) to angular velocities.

    :param q: A unit quaternion represented as a 4D array [w, x, y, z].
    :return: A 4x3 attitude Jacobian matrix.
    :ref: https://rexlab.ri.cmu.edu/papers/planning_with_attitude.pdf
    """
    w, v = q[0], q[1:]
    return jnp.vstack([-v.T, jnp.eye(3) * w + skew_symmetric(v)])


def jac_dq2v(model: mjx.Model, q: jnp.ndarray):
    """
    Compute the Jacobian matrix for converting from generalized positions to velocities.

    This function calculates the Jacobian matrix that maps changes in generalized
    positions (q) to generalized velocities (v) for a given MuJoCo model. It handles
    different joint types, including free joints, ball joints, and other types.

    :param model: A MuJoCo model object (mjx.Model).
    :param q: The current generalized positions of the model.
    :return: A Jacobian matrix of shape (nq, nv), where nq is the number of position
             variables and nv is the number of velocity variables.
    """
    jac = jnp.zeros((model.nq, model.nv))

    row_idx, col_idx = 0, 0
    for jnt_id in range(model.njnt):
        jnt_type = model.jnt_type[jnt_id]
        jnt_qpos_idx_begin = model.jnt_qposadr[jnt_id]
        match jnt_type:
            case mj.mjtJoint.mjJNT_FREE:
                jac = jac.at[row_idx : row_idx + 3, col_idx : col_idx + 3].set(jnp.eye(3))
                jac = jac.at[row_idx + 3 : row_idx + 7, col_idx + 3 : col_idx + 6].set(
                    attitude_jacobian(q[jnt_qpos_idx_begin + 3 : jnt_qpos_idx_begin + 7])
                )
                row_idx += 7
                col_idx += 6
            case mj.mjtJoint.mjJNT_BALL:
                jac = jac.at[row_idx : row_idx + 4, col_idx : col_idx + 3].set(
                    attitude_jacobian(q[jnt_qpos_idx_begin : jnt_qpos_idx_begin + 4])
                )
                row_idx += 4
                col_idx += 3
            case _:
                jac = jac.at[row_idx, col_idx].set(1)
                row_idx += 1
                col_idx += 1
    return jac
