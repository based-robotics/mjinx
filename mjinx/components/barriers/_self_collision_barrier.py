from __future__ import annotations

import warnings
from enum import Enum
from typing import Callable, Sequence, final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.barriers import Barrier, JaxBarrier
from mjinx.configuration import get_distance, sorted_pair
from mjinx.typing import ArrayOrFloat, CollisionBody, CollisionPair


@jdc.pytree_dataclass
class JaxSelfCollisionBarrier(JaxBarrier):
    r"""..."""

    d_min_vec: jnp.ndarray
    collision_pairs: jdc.Static[list[CollisionPair]]

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return get_distance(self.model, data, self.collision_pairs) - self.d_min_vec


class SelfCollisionBarrier(Barrier[JaxSelfCollisionBarrier]):
    JaxComponentType: type = JaxSelfCollisionBarrier
    d_min: float
    collision_bodies: Sequence[CollisionBody]
    exclude_collisions: set[CollisionPair]
    collision_pairs: list[CollisionPair]

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        d_min: ArrayOrFloat = 0,
        collision_bodies: Sequence[CollisionBody] = (),
        excluded_collisions: Sequence[tuple[CollisionBody, CollisionBody]] = (),
    ):
        self.collision_bodies = collision_bodies
        self.__exclude_collisions_raw: Sequence[tuple[CollisionBody, CollisionBody]] = excluded_collisions

        super().__init__(name, gain, gain_fn, safe_displacement_gain)
        self.d_min = d_min

    def __validate_body_pair(self, body1_id: int, body2_id: int) -> bool:
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

    def __validate_geom_pair(self, geom1_id: int, geom2_id: int) -> bool:
        # ref: https://mujoco.readthedocs.io/en/stable/computation/index.html#selection
        return (
            self.model.geom_contype[geom1_id] & self.model.geom_conaffinity[geom2_id]
            or self.model.geom_contype[geom2_id] & self.model.geom_conaffinity[geom1_id]
        )

    def __body2id(self, body: CollisionBody):
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
        collision_bodies: Sequence[CollisionBody] = (),
        excluded_collisions: set[CollisionPair] = (),
    ) -> list[CollisionPair]:
        collision_pairs: set[CollisionPair] = set()
        for i in range(len(collision_bodies)):
            for k in range(i + 1, len(collision_bodies)):
                body1_id = self.__body2id(collision_bodies[i])
                body2_id = self.__body2id(collision_bodies[k])

                if (
                    body1_id == body2_id  # If bodies are the same (somehow),
                    or sorted_pair(body1_id, body2_id) in excluded_collisions  # or body pair is excluded,
                    or not self.__validate_body_pair(body1_id, body2_id)  # or body pair is not valid for other reason
                ):
                    # then skip
                    continue

                body1_geom_start = self.model.body_geomadr[body1_id]
                body1_geom_end = body1_geom_start + self.model.body_geomnum[body1_id]

                body2_geom_start = self.model.body_geomadr[body2_id]
                body2_geom_end = body2_geom_start + self.model.body_geomnum[body2_id]

                for body1_geom_i in range(body1_geom_start, body1_geom_end):
                    for body2_geom_i in range(body2_geom_start, body2_geom_end):
                        if self.__validate_geom_pair(body1_geom_i, body2_geom_i):
                            collision_pairs.add(sorted_pair(body1_geom_i, body2_geom_i))

        return list(collision_pairs)

    def update_model(self, model: mjx.Model):
        super().update_model(model)
        if len(self.collision_bodies) == 0:
            self.collision_bodies = list(range(self.model.nbody))

        self.exclude_collisions: set[CollisionPair] = {
            sorted_pair(
                self.__body2id(body1),
                self.__body2id(body2),
            )
            for body1, body2 in self.__exclude_collisions_raw
        }
        self.collision_pairs = self._generate_collision_pairs(
            self.collision_bodies,
            self.exclude_collisions,
        )
        self._dim = len(self.collision_pairs)

    @property
    def d_min_vec(self) -> jnp.ndarray:
        if self._dim == -1:
            raise ValueError(
                "fail to calculate d_min without dimension specified. "
                "You should provide robot model or pass component into the problem first."
            )

        return jnp.ones(self.dim) * self.d_min
