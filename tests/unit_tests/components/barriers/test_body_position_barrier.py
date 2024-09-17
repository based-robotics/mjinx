from __future__ import annotations

import warnings
from enum import Enum
from typing import Callable, Sequence, final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components.barriers._body_barrier import BodyBarrier, JaxBodyBarrier
from mjinx.typing import ArrayOrFloat


class PositionLimitType(Enum):
    MIN = 0
    MAX = 1
    BOTH = 2

    @staticmethod
    def from_str(type: str) -> PositionLimitType:
        match type.lower():
            case "min":
                return PositionLimitType.MIN
            case "max":
                return PositionLimitType.MAX
            case "both":
                return PositionLimitType.BOTH
            case _:
                raise ValueError(
                    f"[PositionLimitType] invalid position limit type: {type}. " f"Expected {{'min', 'max', 'both'}}"
                )

    @staticmethod
    def includes_min(type: PositionLimitType) -> bool:
        return type == PositionLimitType.MIN or type == PositionLimitType.BOTH

    @staticmethod
    def includes_max(type: PositionLimitType) -> bool:
        return type == PositionLimitType.MAX or type == PositionLimitType.BOTH


@jdc.pytree_dataclass
class JaxPositionBarrier(JaxBodyBarrier):
    r"""..."""

    p_min: jnp.ndarray
    p_max: jnp.ndarray

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return jnp.concatenate(
            [
                data.xpos[self.body_id, self.mask_idxs] - self.p_min,
                self.p_max - data.xpos[self.body_id, self.mask_idxs],
            ]
        )


class PositionBarrier(BodyBarrier[JaxPositionBarrier]):
    __p_min: jnp.ndarray
    __p_max: jnp.ndarray

    __limit_type: PositionLimitType

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        body_name: str,
        p_min: ArrayOrFloat | None = None,
        p_max: ArrayOrFloat | None = None,
        limit_type: str = "both",
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        mask: Sequence | None = None,
    ):
        mask = mask if mask is not None else jnp.array([1, 1, 1])
        super().__init__(name, gain, body_name, gain_fn, safe_displacement_gain, mask)
        if limit_type not in {"min", "max", "both"}:
            raise ValueError("[PositionBarrier] PositionBarrier.limit should be either 'min', 'max', or 'both'")

        # Setting up the dimension, using mask and limit type
        self.__limit_type = PositionLimitType.from_str(limit_type)
        n_axes = len(self.mask_idxs)
        self._dim = 2 * n_axes if self.limit_type == PositionLimitType.BOTH else n_axes

        self.update_p_min(
            p_min if p_min is not None else jnp.empty(len(self.mask_idxs)),
            ignore_warnings=True,
        )
        self.update_p_max(
            p_max if p_max is not None else jnp.empty(len(self.mask_idxs)),
            ignore_warnings=True,
        )

    @property
    def limit_type(self) -> PositionLimitType:
        return self.__limit_type

    @property
    def p_min(self) -> jnp.ndarray:
        return self.__p_min

    @p_min.setter
    def p_min(self, value: ArrayOrFloat):
        self.update_p_min(value)

    def update_p_min(self, p_min: ArrayOrFloat, ignore_warnings: bool = False):
        if isinstance(p_min, float):
            p_min = jnp.ones(len(self.mask_idxs)) * p_min

        if p_min.shape[-1] != len(self.mask_idxs):
            raise ValueError(
                f"[PositionBarrier] wrong dimension of p_min: expected {len(self.mask_idxs)}, got {p_min.shape[-1]}"
            )
        if not ignore_warnings and not PositionLimitType.includes_min(self.limit_type):
            warnings.warn(
                "[PositionBarrier] type of the limit does not include minimum bound. Assignment is ignored",
                stacklevel=2,
            )
            return

        self._modified = True
        self.__p_min = p_min if isinstance(p_min, jnp.ndarray) else jnp.array(p_min)

    @property
    def p_max(self) -> jnp.ndarray:
        return self.__p_max

    @p_max.setter
    def p_max(self, value: ArrayOrFloat):
        self.update_p_max(value)

    def update_p_max(self, p_max: ArrayOrFloat, ignore_warnings: bool = False):
        if isinstance(p_max, float):
            p_max = jnp.ones(len(self.mask_idxs)) * p_max

        if p_max.shape[-1] != len(self.mask_idxs):
            raise ValueError(
                f"[PositionBarrier] wrong dimension of p_max: expected {len(self.mask_idxs)}, got {p_max.shape[-1]}"
            )
        if not ignore_warnings and not PositionLimitType.includes_max(self.limit_type):
            warnings.warn(
                "[PositionBarrier] type of the limit does not include maximum bound. Assignment is ignored",
                stacklevel=2,
            )
            return
        self._modified = True
        self.__p_max = p_max if isinstance(p_max, jnp.ndarray) else jnp.array(p_max)

    def _build_component(self) -> JaxPositionBarrier:
        return JaxPositionBarrier(
            dim=self.dim,
            model=self.model,
            gain=self.vector_gain,
            gain_function=self.gain_fn,
            safe_displacement_gain=self.safe_displacement_gain,
            body_id=self.body_id,
            p_min=self.p_min,
            p_max=self.p_max,
            mask_idxs=self.mask_idxs,
        )
