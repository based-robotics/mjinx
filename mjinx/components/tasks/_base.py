from typing import Callable, final, override

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components import Component, JaxComponent
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass(kw_only=True)
class JaxTask(JaxComponent):

    cost: jnp.ndarray
    lm_damping: jdc.Static[float]

    def compute_error(self, data: mjx.Data) -> jnp.ndarray:
        return self.__call__(data)

    @override
    @final
    def compute_qp_objective(
        self,
        data: mjx.Data,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""..."""
        jacobian = self.compute_jacobian(data)
        minus_gain_error = -self.gain * jax.lax.map(self.gain_function, self.compute_error(data))

        weighted_jacobian = self.cost.T @ jacobian  # [cost]
        weighted_error = self.cost.T @ minus_gain_error  # [cost]

        mu = self.lm_damping * weighted_error.T @ weighted_error  # [cost]^2
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
    __cost: jnp.ndarray
    __cost_raw: jnp.ndarray

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
    ):
        super().__init__(name, gain, gain_fn)
        self.__lm_damping = lm_damping

        self.cost = cost

    @property
    def cost(self) -> jnp.ndarray:
        return self.__cost

    @property
    def matrix_cost(self) -> jnp.ndarray:
        # scalar -> jnp.ones(self.dim) * scalar
        # vector -> jnp.diag(vector)
        # matrix -> matrix
        if self._dim == -1:
            raise ValueError(
                "fail to calculate matrix cost without dimension specified. "
                "You should provide robot model or pass component into the problem first."
            )

        match self.cost.ndim:
            case 0:
                return jnp.eye(self.dim) * self.cost
            case 1:
                if len(self.gain) != self.dim:
                    raise ValueError(
                        f"fail to construct matrix {self.dim}x{self.dim} from vector of length {self.gain.shape}"
                    )
                return jnp.diag(self.gain)
            case 2:
                if self.gain.shape != (
                    self.dim,
                    self.dim,
                ):
                    raise ValueError(f"wrong shape of the gain: {self.gain.shape} != ({self.dim}, {self.dim},)")
                return self.gain
            case _:
                raise ValueError("fail to construct matrix cost from cost with ndim > 2")

    @cost.setter
    def cost(self, value: ArrayOrFloat):
        self.update_cost(value)

    def update_cost(self, cost: ArrayOrFloat):
        self._modified = True
        self.__cost = cost if isinstance(cost, jnp.ndarray) else jnp.array(cost)

    @property
    def lm_damping(self) -> float:
        return self.__lm_damping
