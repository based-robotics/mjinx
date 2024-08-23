import abc
from dataclasses import field
from typing import Callable, ClassVar, Self

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from ..configuration import update


@jdc.pytree_dataclass(kw_only=True)
class Barrier(abc.ABC):
    r"""..."""

    dim: ClassVar[int]

    model: mjx.Model
    gain: jnp.ndarray
    gain_function: jdc.Static[Callable[[float], float]] = field(default_factory=lambda: (lambda x: x))
    safe_displacement_gain: float = 0.0

    def copy_and_set(self, **kwargs) -> Self:
        r"""..."""
        new_args = self.__dict__ | kwargs
        return self.__class__(**new_args)

    @abc.abstractmethod
    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray: ...  # h(q) > 0!

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        return jax.jacrev(
            lambda q, model=self.model: self.compute_barrier(
                update(model, q=q),
            ),
            argnums=0,
        )(data.qpos)

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
