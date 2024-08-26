import warnings
from enum import Enum
from typing import Callable, Self, override

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.barriers.body_barriers.body_barrier import BodyBarrier, JaxBodyBarrier
from mjinx.typing import Gain


class PositionLimitType(Enum):
    MIN = 0
    MAX = 1
    BOTH = 2

    @staticmethod
    def from_str(type: str) -> Self:
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
    def includes_min(type: Self) -> bool:
        return type == PositionLimitType.MIN or type == PositionLimitType.BOTH

    @staticmethod
    def incldues_max(type: Self) -> bool:
        return type == PositionLimitType.MAX or type == PositionLimitType.BOTH


@jdc.pytree_dataclass
class JaxPositionBarrier(JaxBodyBarrier):
    r"""..."""

    p_min: jnp.ndarray
    p_max: jnp.ndarray

    min_axes: jdc.Static[tuple[int, ...]]
    max_axes: jdc.Static[tuple[int, ...]]

    def __post_init__(self):
        object.__setattr__(self, "gain", jnp.concatenate([self.position_gain, self.position_gain]))

    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray:
        return jnp.concatenate(
            [
                data.xpos[self.body_id, self.min_axes] - self.p_min,
                self.p_max - data.xpos[self.body_id, self.min_axes],
            ]
        )


class PositionBarrier(BodyBarrier[JaxPositionBarrier]):
    __p_min: jnp.ndarray
    __p_max: jnp.ndarray

    __limit_type: PositionLimitType
    __axes_str: str
    __axes_idx: tuple[int, ...]

    def __init__(
        self,
        gain: Gain,
        body_name: str,
        p_min: jnp.ndarray | np.ndarray | None = None,
        p_max: jnp.ndarray | np.ndarray | None = None,
        limit_type: str = "both",
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        axes: str = "xyz",
    ):
        super().__init__(gain, body_name, gain_fn, safe_displacement_gain)
        if limit_type not in {"min", "max", "both"}:
            raise ValueError("[PositionBarrier] PositionBarrier.limit should be either 'min', 'max', or 'both'")

        self.__limit_type = PositionLimitType.from_str(limit_type)
        self.__axes_str = axes
        self.__axes_idx = tuple([i for i in range(3) if "xyz"[i] in self.axes])

        self.__p_min = p_min if p_min is not None else jnp.empty(0)
        self.__p_max = p_max if p_max is not None else jnp.empty(0)

    @property
    def limit_type(self) -> PositionLimitType:
        return self.limit_type

    @property
    def axes(self) -> str:
        return self.__axes_str

    @property
    def p_min(self) -> jnp.ndarray:
        return self.__p_min

    @p_min.setter
    def p_min(self, value: np.ndarray | jnp.ndarray):
        self.update_p_min(value)

    def update_p_min(self, p_min: np.ndarray | jnp.ndarray):
        if p_min.shape[-1] != len(self.axes):
            raise ValueError(
                f"[PositionBarrier] wrong dimension of p_min: expected {len(self.axes)}, got {p_min.shape[-1]}"
            )
        if not PositionLimitType.incldues_min(self.limit_type):
            warnings.warn(
                "[PositionBarrier] type of the limit does not include minimum bound. Assignment is ignored",
                stacklevel=2,
            )
            return

        self._modified = True
        self.__p_min = p_min if isinstance(p_min, jnp.ndarray) else jnp.ndarray(p_min)

    @property
    def p_max(self) -> jnp.ndarray:
        return self.__p_max

    @p_max.setter
    def p_max(self, value: np.ndarray | jnp.ndarray):
        self.update_p_max(value)

    def update_p_max(self, p_max: np.ndarray | jnp.ndarray):
        if p_max.shape[-1] != len(self.axes):
            raise ValueError(
                f"[PositionBarrier] wrong dimension of p_max: expected {len(self.axes)}, got {p_max.shape[-1]}"
            )
        if not PositionLimitType.incldues_max(self.limit_type):
            warnings.warn(
                "[PositionBarrier] type of the limit does not include maximum bound. Assignment is ignored",
                stacklevel=2,
            )
            return
        self._modified = True
        self.__p_max = p_max if isinstance(p_max, jnp.ndarray) else jnp.ndarray(p_max)

    @override
    def _build_component(self) -> JaxPositionBarrier:
        return JaxPositionBarrier(
            dim=6 if self.limit_type == PositionLimitType.BOTH else 3,
            model=self.model,
            gain_function=self.gain_fn,
            safe_displacement_gain=self.safe_displacement_gain,
            body_id=self.body_id,
            p_min=self.p_min,
            p_max=self.p_max,
            min_axes=self.__axes_idx if PositionLimitType.incldues_min(self.limit_type) else (),
            max_azes=self.__axes_idx if PositionLimitType.incldues_max(self.limit_type) else (),
        )
