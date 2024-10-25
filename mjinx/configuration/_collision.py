import jax.numpy as jnp
import mujoco as mj
from mujoco import mjx

from mjinx.typing import CollisionPair


def sorted_pair(x: int, y: int) -> tuple[int, int]:
    """
    Return a sorted pair of integers.

    :param x: The first integer.
    :param y: The second integer.
    :return: A tuple of the two integers, sorted in ascending order.
    """
    return (min(x, y), max(x, y))


def get_distance(
    model: mjx.Model,
    data: mjx.Data,
    collision_pairs: list[CollisionPair],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute the distances for the given collision pairs.

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
