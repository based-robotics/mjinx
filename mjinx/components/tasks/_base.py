from typing import Callable, final, override

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components import Component, JaxComponent
from mjinx.typing import Gain


@jdc.pytree_dataclass(kw_only=True)
class JaxTask(JaxComponent):

    lm_damping: jdc.Static[float]

    def compute_error(self, data: mjx.Data):
        self.__call__(data)

    @override
    @final
    def compute_qp_objective(
        self,
        data: mjx.Data,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""..."""
        jacobian = self.compute_jacobian(data)
        minus_gain_error = -self.gain * jax.lax.scan(self.gain_function, self.compute_error(data))

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

    @override
    @final
    def compute_qp_inequality(self, data: mjx.Data) -> tuple[jnp.ndarray, jnp.ndarray]:
        # TODO: make sure it is handled well
        return tuple(jnp.empty(0), jnp.empty(0))


class Task[T: JaxTask](Component[T]):
    __lm_damping: float

    def __init__(self, gain: Gain, gain_fn: Callable[[float], float] | None = None, lm_damping: float = 0):
        super().__init__(gain, gain_fn)
        self.__lm_damping = lm_damping

    @property
    def lm_damping(self) -> float:
        return self.__lm_damping
