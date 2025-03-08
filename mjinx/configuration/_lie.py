import jax.numpy as jnp
import jaxlie
import mujoco as mj
from jaxlie import SE3, SO3
from mujoco import mjx


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
