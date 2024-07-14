import abc
from dataclasses import InitVar, field
from typing import ClassVar, Self

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
from jax_dataclasses._copy_and_mutate import _Mutability as Mutability


@jdc.pytree_dataclass(kw_only=True)
class Task(abc.ABC):
    dim: ClassVar[int]

    model: mjx.Model
    cost: jnp.ndarray
    gain: jnp.ndarray
    lm_damping: jdc.Static[float] = 0.0

    def copy_and_set(self, **kwargs) -> Self:
        r"""..."""
        new_args = self.__dict__ | kwargs
        return self.__class__(**new_args)

    @abc.abstractmethod
    def compute_error(self, data: mjx.Data) -> jnp.ndarray:
        r"""..."""

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        return jax.jacrev(
            lambda q, model=self.model, data=data: self.compute_error(
                model,
                data.replace(qpos=q),
            ),
            argnums=0,
        )(data.qpos)

    def compute_qp_objective(
        self,
        data: mjx.Data,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""..."""
        jacobian = self.compute_jacobian(data)
        minus_gain_error = -self.gain * self.compute_error(data)

        weighted_jacobian = self.cost @ jacobian  # [cost]
        weighted_error = self.cost @ minus_gain_error  # [cost]

        mu = self.lm_damping * weighted_error @ weighted_error  # [cost]^2
        # TODO: nv is a dimension of the tangent space, right?..
        eye_tg = jnp.eye(self.model.nv)
        # Our Levenberg-Marquardt damping `mu * eye_tg` is isotropic in the
        # robot's tangent space. If it helps we can add a tangent-space scaling
        # to damp the floating base differently from joint angular velocities.
        H = weighted_jacobian.T @ weighted_jacobian + mu * eye_tg
        c = -weighted_error.T @ weighted_jacobian
        return (H, c)
