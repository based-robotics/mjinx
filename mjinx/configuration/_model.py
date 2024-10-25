import jax
import jax.numpy as jnp
import mujoco as mj
from jaxlie import SE3, SO3
from mujoco import mjx


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


def get_frame_jacobian_local(
    model: mjx.Model, data: mjx.Data, obj_id: int, obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY
) -> jax.Array:
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


def body_point_jacobian(
    model: mjx.Model, data: mjx.Data, point: jnp.ndarray, body_id: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    jacobian = mjx._src.support.jac(model, data, point, body_id)
    print(jacobian[0].shape, jacobian[1].shape)
    return jacobian
