from __future__ import annotations

import warnings
from enum import Enum
from typing import Callable, Sequence, final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components.barriers import Barrier, JaxBarrier
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxSelfCollisionBarrier(JaxBarrier):
    r"""..."""

    d_min_vec: jnp.ndarray

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return data.contact.dist[self.mask_idxs] - self.d_min_vec


class SelfCollisionBarrier(Barrier[JaxSelfCollisionBarrier]):
    d_min_exceptions: list[tuple[int, float]]
    __d_min: jnp.ndarray

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        d_min: ArrayOrFloat = 0,
        mask: Sequence | None = None,
    ):
        super().__init__(name, gain, gain_fn, safe_displacement_gain, mask)
        self.d_min = d_min

    @property
    def d_min(self) -> jnp.ndarray:
        return self.__d_min

    @d_min.setter
    def d_min(self, value: ArrayOrFloat):
        self.update_d_min(value)

    def update_d_min(self, d_min: ArrayOrFloat):
        d_min = jnp.array(d_min)

    @property
    def d_min_vector(self) -> jnp.ndarray:
        # scalar -> jnp.ones(self.dim) * scalar
        # vector -> vector
        if self._dim == -1:
            raise ValueError(
                "fail to calculate d_min without dimension specified. "
                "You should provide robot model or pass component into the problem first."
            )

        match self.d_min.ndim:
            case 0:
                return jnp.ones(self.dim) * self.d_min
            case 1:
                if len(self.d_min) != self.dim:
                    raise ValueError(f"invalid d_min size: {self.d_min.shape} != {self.dim}")
                return self.d_min
            case _:  # pragma: no cover
                raise ValueError("fail to construct vector from d_min with ndim > 1")

    def _build_component(self) -> JaxSelfCollisionBarrier:
        return JaxSelfCollisionBarrier(
            dim=self.dim,
            model=self.model,
            vector_gain=self.gain,
            gain_fn=self.gain_fn,
            mask_idxs=self.mask_idxs,
            safe_displacement_gain=self.safe_displacement_gain,
            d_min_vec=self.d_min_vec,
        )
