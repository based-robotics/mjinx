import abc

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx


@jdc.pytree_dataclass
class Task(abc.ABC):
    # Default values and optional arguments
    cost: jnp.ndarray
    gain: float
    lm_damping: float

    @abc.abstractmethod
    def compute_error(self, model: mjx.Model, data: mjx.Data) -> jnp.ndarray:
        r"""..."""

    @abc.abstractmethod
    def compute_jacobian(self, model: mjx.Model, data: mjx.Data) -> jnp.ndarray:
        r"""..."""

    def compute_qp_objective(self, model: mjx.Model, data: mjx.Data) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""..."""
        jacobian = self.compute_jacobian(model, data)
        minus_gain_error = -self.gain * self.compute_error(model, data)

        weighted_jacobian = self.cost @ jacobian  # [cost]
        weighted_error = self.cost @ minus_gain_error  # [cost]
        # mu = self.lm_damping * weighted_error @ weighted_error  # [cost]^2

        # TODO: handle Levenberg-Marquardt damping
        # eye_tg = configuration.tangent.eye
        # Our Levenberg-Marquardt damping `mu * eye_tg` is isotropic in the
        # robot's tangent space. If it helps we can add a tangent-space scaling
        # to damp the floating base differently from joint angular velocities.
        H = weighted_jacobian.T @ weighted_jacobian  # + mu * eye_tg
        c = -weighted_error.T @ weighted_jacobian
        return (H, c)

    def __repr__(self):
        """..."""
        return f"Task(" f"cost={self.cost}, " f"gain={self.gain}, " f"lm_damping={self.lm_damping})"
