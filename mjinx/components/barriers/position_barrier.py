from dataclasses import field

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from ..configuration import joint_difference
from .base import Barrier


@jdc.pytree_dataclass
class PositionBarrier(Barrier):
    r"""..."""

    dim = 6

    frame_id: int
    p_min: jnp.ndarray
    p_max: jnp.ndarray
    gain: jnp.ndarray = field(init=False)
    position_gain: jnp.ndarray

    def __post_init__(self):
        object.__setattr__(self, "gain", jnp.concatenate([self.position_gain, self.position_gain]))

    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray:
        return jnp.concatenate(
            [
                data.xpos[self.frame_id] - self.p_min,
                self.p_max - data.xpos[self.frame_id],
            ]
        )


@jdc.pytree_dataclass
class PositionLowerBarrier(Barrier):
    r"""..."""

    dim = 3

    frame_id: int
    p_min: jnp.ndarray

    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray:
        return data.xpos[self.frame_id] - self.p_min


@jdc.pytree_dataclass
class PositionUpperBarrier(Barrier):
    r"""..."""

    frame_id: int
    p_max: jnp.ndarray
    axes: jdc.Static[str] = field(default=lambda: "xyz")
    _p_idx: jdc.Static[tuple[int, ...]] = field(init=False)

    def __post_init__(self):
        _p = [i for i in range(3) if "xyz"[i] in self.axes]
        object.__setattr__(
            self,
            "_p_idx",
            _p,
        )
        object.__setattr__(self, "dim", len(self._p_idx))

    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray:
        return self.p_max - data.xpos[self.frame_id, self._p_idx]
