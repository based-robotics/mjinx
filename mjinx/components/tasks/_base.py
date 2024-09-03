from typing import Callable

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import numpy as np

from mjinx.components import Component, JaxComponent
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxTask(JaxComponent):

    cost: jnp.ndarray
    lm_damping: jdc.Static[float]

    def compute_error(self, data: mjx.Data) -> jnp.ndarray:
        return self.__call__(data)


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
        mask: jnp.ndarray | np.ndarray | None = None,
    ):
        super().__init__(name, gain, gain_fn, mask)
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
