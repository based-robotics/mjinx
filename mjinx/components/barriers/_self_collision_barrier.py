from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import final

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.barriers import Barrier, JaxBarrier
from mjinx.configuration import body_point_jacobian, compute_collision_pairs, sorted_pair
from mjinx.typing import ArrayOrFloat, CollisionBody, CollisionPair


@jdc.pytree_dataclass
class JaxSelfCollisionBarrier(JaxBarrier):
    """
    A JAX implementation of a self-collision barrier function.

    This class extends JaxBarrier to provide barrier functions that prevent
    self-collisions between different parts of the robot.

    :param d_min_vec: The minimum allowed distances between collision pairs.
    :param collision_pairs: A list of collision pairs to check.
    """

    d_min_vec: jnp.ndarray
    collision_pairs: jdc.Static[list[CollisionPair]]
    n_closest_pairs: jdc.Static[int]

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the self-collision barrier value.

        :param data: The MuJoCo simulation data.
        :return: The computed self-collision barrier value.
        """
        dists = compute_collision_pairs(self.model, data, self.collision_pairs).dist
        return -jax.lax.top_k(-dists, self.n_closest_pairs)[0] - self.d_min_vec

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the Jacobian of the barrier function with respect to joint positions.

        This method implements an analytical Jacobian computation which is more efficient
        than autodifferentiation. It computes the Jacobian by calculating how changes in
        joint positions affect the distances between collision pairs.

        :param data: The MuJoCo simulation data containing the current state of the system.
        :return: The Jacobian matrix of shape (n_collision_pairs, n_joints) where each entry (i,j)
                represents how the i-th collision distance changes with respect to the j-th joint position.
        """

        def jac_row(
            dist: jnp.ndarray,  # Scalar distance between collision pair
            point: jnp.ndarray,  # (3,) array of contact point
            normal: jnp.ndarray,  # (3,) array of contact normal
            body_id1: jnp.ndarray,  # Scalar body index for first body
            body_id2: jnp.ndarray,  # Scalar body index for second body
        ) -> jnp.ndarray:
            """
            Compute a single row of the Jacobian matrix for one collision pair.
            """
            p1 = point - jnp.repeat(dist, 3) * normal / 2
            p2 = point + jnp.repeat(dist, 3) * normal / 2
            p_jac1 = body_point_jacobian(self.model, data, p1, body_id1)[0]
            p_jac2 = body_point_jacobian(self.model, data, p2, body_id2)[0]
            return normal @ (p_jac2 - p_jac1).T

        collisions = compute_collision_pairs(self.model, data, self.collision_pairs)
        col_bodies = self.model.geom_bodyid[self.collision_pairs]
        jac = jax.vmap(jac_row)(
            collisions.dist, collisions.pos, collisions.frame[:, 2], col_bodies[:, 0], col_bodies[:, 1]
        )
        return jac


class SelfCollisionBarrier(Barrier[JaxSelfCollisionBarrier]):
    """
    A self-collision barrier class that wraps the JAX self-collision barrier implementation.

    This class provides a high-level interface for self-collision barrier functions.

    :param name: The name of the barrier.
    :param gain: The gain for the barrier function.
    :param gain_fn: A function to compute the gain dynamically.
    :param safe_displacement_gain: The gain for computing safe displacements. Defaults to identity function
    :param d_min: The minimum allowed distance between collision pairs. Defaults to zero.
    :param collision_bodies: A sequence of bodies to check for collisions. Defaults to zero.
    :param excluded_collisions: A sequence of body pairs to exclude from collision checking.
        Defaults to no excluded pairs.
    :param n_closest_pairs: amount of closest pairs to consider. Defaults to all pairs considered.
    """

    JaxComponentType: type = JaxSelfCollisionBarrier
    d_min: float
    collision_bodies: Sequence[CollisionBody]
    exclude_collisions: set[CollisionPair]
    collision_pairs: list[CollisionPair]
    n_closest_pairs: int

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        d_min: float = 0,
        collision_bodies: Sequence[CollisionBody] = (),
        excluded_collisions: Sequence[tuple[CollisionBody, CollisionBody]] = (),
        n_closest_pairs: int = -1,
    ):
        self.collision_bodies = collision_bodies
        self.n_closest_pairs = n_closest_pairs
        self.__exclude_collisions_raw: Sequence[tuple[CollisionBody, CollisionBody]] = excluded_collisions

        super().__init__(name, gain, gain_fn, safe_displacement_gain)
        self.d_min = d_min

    def validate_body_pair(self, body1_id: int, body2_id: int) -> bool:
        """
        Validate if a pair of bodies should be considered for collision checking.

        :param body1_id: The ID of the first body.
        :param body2_id: The ID of the second body.
        :return: True if the pair is valid for collision checking, False otherwise.
        """
        # body_weldid is the ID of the body's weld.
        body_weldid1 = self.model.body_weldid[body1_id]
        body_weldid2 = self.model.body_weldid[body2_id]

        # weld_parent_id is the ID of the parent of the body's weld.
        weld_parent_id1 = self.model.body_parentid[body_weldid1]
        weld_parent_id2 = self.model.body_parentid[body_weldid2]

        # weld_parent_weldid is the weld ID of the parent of the body's weld.
        weld_parent_weldid1 = self.model.body_weldid[weld_parent_id1]
        weld_parent_weldid2 = self.model.body_weldid[weld_parent_id2]

        is_parent_child = body_weldid1 == weld_parent_weldid2 or body_weldid2 == weld_parent_weldid1

        weld1 = self.model.body_weldid[body1_id]
        weld2 = self.model.body_weldid[body2_id]

        is_welded = weld1 == weld2

        return not (is_parent_child or is_welded)

    def validate_geom_pair(self, geom1_id: int, geom2_id: int) -> bool:
        # ref: https://mujoco.readthedocs.io/en/stable/computation/index.html#selection
        return (
            self.model.geom_contype[geom1_id] & self.model.geom_conaffinity[geom2_id]
            or self.model.geom_contype[geom2_id] & self.model.geom_conaffinity[geom1_id]
        )

    def body2id(self, body: CollisionBody):
        """
        Convert a body identifier to its corresponding ID in the model.

        :param body: The body identifier (either an integer ID or a string name).
        :return: The integer ID of the body.
        :raises ValueError: If the body identifier is invalid.
        """
        if isinstance(body, int):
            return body
        elif isinstance(body, str):
            return mjx.name2id(
                self.model,
                mj.mjtObj.mjOBJ_BODY,
                body,
            )
        else:
            raise ValueError(f"invalid body type: expected string or int, got {type(body)}")

    def _generate_collision_pairs(
        self,
        collision_bodies: Sequence[CollisionBody],
        excluded_collisions: set[CollisionPair],
    ) -> list[CollisionPair]:
        """Construct colliison bodies, based on the model, their list.

        The names from the list are used pairwise among each other.

        :param collision_bodies: List of several bodies.
        :param excluded_collisions: set of excluded collision pairs.
        """
        if excluded_collisions is None:
            excluded_collisions = set()

        excluded_collisions_set = set(excluded_collisions)
        collision_pairs: set[CollisionPair] = set()
        for i in range(len(collision_bodies)):
            for k in range(i + 1, len(collision_bodies)):
                body1_id = self.body2id(collision_bodies[i])
                body2_id = self.body2id(collision_bodies[k])

                if (
                    body1_id == body2_id  # If bodies are the same (somehow),
                    or sorted_pair(body1_id, body2_id) in excluded_collisions_set  # or body pair is excluded,
                    or not self.validate_body_pair(body1_id, body2_id)  # or body pair is not valid for other reason
                ):
                    # then skip
                    continue

                body1_geom_start = self.model.body_geomadr[body1_id]
                body1_geom_end = body1_geom_start + self.model.body_geomnum[body1_id]

                body2_geom_start = self.model.body_geomadr[body2_id]
                body2_geom_end = body2_geom_start + self.model.body_geomnum[body2_id]

                for body1_geom_i in range(body1_geom_start, body1_geom_end):
                    for body2_geom_i in range(body2_geom_start, body2_geom_end):
                        if self.validate_geom_pair(body1_geom_i, body2_geom_i):
                            collision_pairs.add(sorted_pair(body1_geom_i, body2_geom_i))

        return list(collision_pairs)

    def update_model(self, model: mjx.Model):
        """
        Update the model and generate collision pairs.

        :param model: The MuJoCo model.
        """
        super().update_model(model)
        if len(self.collision_bodies) == 0:
            self.collision_bodies = list(range(self.model.nbody))

        self.exclude_collisions: set[CollisionPair] = {
            sorted_pair(
                self.body2id(body1),
                self.body2id(body2),
            )
            for body1, body2 in self.__exclude_collisions_raw
        }
        self.collision_pairs = self._generate_collision_pairs(
            self.collision_bodies,
            self.exclude_collisions,
        )
        if self.n_closest_pairs == -1:
            self.n_closest_pairs = len(self.collision_pairs)
        self._dim = self.n_closest_pairs

    @property
    def d_min_vec(self) -> jnp.ndarray:
        """
        Get the vector of minimum allowed distances for each collision pair.

        :return: An array of minimum distances.
        :raises ValueError: If the dimension is not set.
        """
        if self._dim == -1:
            raise ValueError(
                "fail to calculate d_min without dimension specified. "
                "You should provide robot model or pass component into the problem first."
            )

        return jnp.ones(self.dim) * self.d_min
