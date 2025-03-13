# Disclaimer: This function is adapted from MuJoCo mjx
# Link: https://github.com/google-deepmind/mujoco/blob/main/mjx/mujoco/mjx/_src/collision_driver.py

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np
from mujoco import mjx
from mujoco.mjx._src.collision_driver import _COLLISION_FUNC
from mujoco.mjx._src.collision_types import FunctionKey

from mjinx.typing import CollisionPair, SimplifiedContact


def sorted_pair(x: int, y: int) -> tuple[int, int]:
    """
    Return a sorted pair of integers.

    :param x: The first integer.
    :param y: The second integer.
    :return: A tuple of the two integers, sorted in ascending order.
    """
    return (min(x, y), max(x, y))


def geom_groups(
    model: mjx.Model,
    collision_pairs: list[CollisionPair],
) -> dict[FunctionKey, SimplifiedContact]:
    """
    Group geometry pairs by their collision function characteristics.

    Groups collision pairs based on geometry types, mesh properties, and collision
    dimensions. For mesh geometries, convex functions are executed separately for
    each distinct mesh in the model since convex functions require static mesh sizes.

    This function is greatly inspired by mujoco.mjx._src.collision_driver module.

    :param model: The MuJoCo model containing geometry information.
    :param collision_pairs: List of collision pairs to be grouped.
    :return: Dictionary mapping function keys to simplified contact information.
    """

    groups_geoms: dict[FunctionKey, list[tuple[int, int]]] = {}

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

        key = FunctionKey(types, data_ids, condim)

        # TODO: uncomment when hfield is supported
        # if types[0] == mj.mjtGeom.mjGEOM_HFIELD:
        #     # add static grid bounds to the grouping key for hfield collisions
        #     geom_rbound_hfield = model.geom_rbound
        #     nrow, ncol = model.hfield_nrow[data_ids[0]], model.hfield_ncol[data_ids[0]]
        #     xsize, ysize = model.hfield_size[data_ids[0]][:2]
        #     xtick, ytick = (2 * xsize) / (ncol - 1), (2 * ysize) / (nrow - 1)
        #     xbound = int(np.ceil(2 * geom_rbound_hfield[g2] / xtick)) + 1
        #     xbound = min(xbound, ncol)
        #     ybound = int(np.ceil(2 * geom_rbound_hfield[g2] / ytick)) + 1
        #     ybound = min(ybound, nrow)
        #     key = FunctionKey(types, data_ids, condim, (xbound, ybound))

        groups_geoms.setdefault(key, []).append((g1, g2))

    groups_contacts = {
        key: SimplifiedContact(geom=np.array(val), dist=None, pos=None, frame=None) for key, val in groups_geoms.items()
    }
    return groups_contacts


def compute_collision_pairs(
    m: mjx.Model,
    d: mjx.Data,
    collision_pairs: list[CollisionPair],
) -> SimplifiedContact:
    """
    Process and compute collisions between specified geometry pairs.

    For each collision pair, this function computes:
    1. The minimum distance between geometries
    2. The contact point location
    3. The contact normal direction

    The distance computation depends on the geometry types, but generally follows:

    .. math::

        d(G_1, G_2) = \\min_{p_1 \\in G_1, p_2 \\in G_2} \\|p_1 - p_2\\| - (r_1 + r_2)

    where:
        - :math:`G_1, G_2` are the geometries
        - :math:`r_1, r_2` are their respective radii/margins
        - :math:`p_1, p_2` are points on the geometries

    :param m: The MuJoCo model containing simulation parameters.
    :param d: The MuJoCo data containing current state information.
    :param collision_pairs: List of geometry pairs to check for collisions.
    :return: Simplified contact information containing collision results.
    """

    if len(collision_pairs) == 0:
        return SimplifiedContact(geom=np.zeros(0), dist=jnp.zeros(0), pos=jnp.zeros(0), frame=jnp.zeros(0))
    groups = geom_groups(m, collision_pairs)

    # run collision functions on groups
    for key, contact in groups.items():
        # run the collision function specified by the grouping key
        func = _COLLISION_FUNC[key.types]
        ncon = func.ncon  # pytype: disable=attribute-error

        dist, pos, frame = func(m, d, key, contact.geom)
        if ncon > 1:
            # repeat contacts to match the number of collisions returned
            def repeat_fn(x, r=ncon):
                return jnp.repeat(x, r, axis=0)

            contact = jax.tree_util.tree_map(repeat_fn, contact)
        groups[key] = contact.replace(dist=dist, pos=pos, frame=frame)

    # collapse contacts together, ensuring they are grouped by condim
    condim_groups: dict[FunctionKey, list[SimplifiedContact]] = {}
    for key, contact in groups.items():
        condim_groups.setdefault(key.condim, []).append(contact)

    contacts = sum([condim_groups[k] for k in sorted(condim_groups)], [])
    contact = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x), *contacts)

    return contact


def get_distance(
    model: mjx.Model,
    data: mjx.Data,
    collision_pairs: list[CollisionPair],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute the distances for the given collision pairs.

    This function is greatly inspired by mujoco.mjx._src.collision_driver module.

    :param model: The MuJoCo model.
    :param data: The MuJoCo data.
    :param collision_pairs: A list of collision pairs to check.
    :return: An array of distances for each collision pair.
    """
    dists = []
    poses = []
    frames = []
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
        dist, pos, frame = collision_fn(
            model,
            data,
            key,
            jnp.array((g1, g2)).reshape(1, -1),
        )
        dists.append(dist.min())
        poses.append(pos)
        frames.append(frame)
    return jnp.array(dists), jnp.vstack(poses), jnp.vstack(frames)


def geom_point_jacobian(model: mjx.Model, data: mjx.Data, point: jnp.ndarray, body_id: jnp.ndarray) -> jnp.ndarray:
    """Compute the Jacobian of a point on a body with respect to joint velocities.

    This function calculates how a specific point on a body moves in world coordinates
    when joint velocities are applied. For a point p on body b, the Jacobian is:

    .. math::

        J_p = \\begin{bmatrix} J_v \\\\ J_\\omega \\end{bmatrix}

    where:
        - :math:`J_v` relates joint velocities to the linear velocity of the point
        - :math:`J_\\omega` relates joint velocities to the angular velocity at the point

    This Jacobian is essential for many robotics applications, including operational space
    control, collision avoidance, and contact modeling.

    :param model: The MuJoCo model.
    :param data: The MuJoCo data.
    :param point: The 3D coordinates of the point in world frame.
    :param body_id: The ID of the body to which the point is attached.
    :return: The point Jacobian matrix (6×nv).
    """
    jacp, jacr = mjx._src.support.jac(model, data, point, body_id)
    return jnp.vstack((jacp.T, jacr.T)).T
