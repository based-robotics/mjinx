import abc
from typing import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components import Component, JaxComponent
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass(kw_only=True)
class JaxBarrier(JaxComponent):
    r"""..."""

    safe_displacement_gain: float

    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray:
        # h(q) > 0!
        return self.__call__(data)

    def compute_safe_displacement(self, data: mjx.Data) -> jnp.ndarray:
        r""""""
        return jnp.zeros(self.model.nv)

    def compute_qp_objective(self, data: mjx.Data) -> tuple[jnp.ndarray, jnp.ndarray]:
        r""""""
        gain_over_jacobian = self.safe_displacement_gain / jnp.linalg.norm(self.compute_jacobian(data)) ** 2

        return (
            gain_over_jacobian * jnp.eye(self.model.nv),
            -gain_over_jacobian * self.compute_safe_displacement(data),
        )

    def compute_qp_inequality(
        self,
        data: mjx.Data,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r""""""
        barrier = self.compute_barrier(data)
        return (
            -self.compute_jacobian(data),
            self.gain * jax.lax.map(self.gain_function, barrier),
        )


class Barrier[T: JaxBarrier](Component[T]):
    __safe_displacement_gain: float

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
    ):
        super().__init__(name, gain, gain_fn)
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
