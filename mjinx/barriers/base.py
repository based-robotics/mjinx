import abc
from typing import Callable, ClassVar

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx


@jdc.pytree_dataclass
class Barrier(abc.ABC):
    r"""..."""

    dim: jdc.Static[int] = -1

    model: mjx.Model
    gain: jnp.ndarray
    gain_function: jdc.Static[Callable[[float], float]]
    safe_displacement_gain: float

    @abc.abstractmethod
    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray: ...  # h(q) > 0!

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        return jax.jacrev(
            lambda q, model=self.model, data=data: self.compute_barrier(
                model,
                data.replace(qpos=q),
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
        dt: float = 1e-3,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r""""""
        return (
            -self.compute_jacobian(data) / dt,
            jnp.array([self.gain[i] * self.gain_function(self.compute_barrier(data)[i]) for i in range(self.dim)]),
        )
