from typing import Callable, Generic, Sequence, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components import Component, JaxComponent
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxBarrier(JaxComponent):
    r"""..."""

    safe_displacement_gain: float

    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray:
        # h(q) > 0!
        return self.__call__(data)

    def compute_safe_displacement(self, data: mjx.Data) -> jnp.ndarray:
        r""""""
        return jnp.zeros(self.model.nv)


AtomicBarrierType = TypeVar("AtomicBarrierType", bound=JaxBarrier)


class Barrier(Generic[AtomicBarrierType], Component[AtomicBarrierType]):
    __safe_displacement_gain: float

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        mask: Sequence | None = None,
    ):
        super().__init__(name, gain, gain_fn, mask)
        self.safe_displacement_gain = safe_displacement_gain

    @property
    def safe_displacement_gain(self) -> float:
        return self.__safe_displacement_gain

    @safe_displacement_gain.setter
    def safe_displacement_gain(self, value: float):
        self.update_safe_displacement_gain(value)

    def update_safe_displacement_gain(self, safe_displacement_gain: float):
        self._modified = True
        self.__safe_displacement_gain = safe_displacement_gain
