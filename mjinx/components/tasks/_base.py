from typing import Callable, Generic, Sequence, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components._base import Component, JaxComponent
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxTask(JaxComponent):
    cost: jnp.ndarray
    lm_damping: jdc.Static[float]

    def compute_error(self, data: mjx.Data) -> jnp.ndarray:
        return self.__call__(data)


AtomicTaskType = TypeVar("AtomicTaskType", bound=JaxTask)


class Task(Generic[AtomicTaskType], Component[AtomicTaskType]):
    lm_damping: float
    __cost: jnp.ndarray

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Sequence | None = None,
    ):
        super().__init__(name, gain, gain_fn, mask)
        if lm_damping < 0:
            raise ValueError("lm_damping has to be positive")
        self.lm_damping = lm_damping

        self.update_cost(cost)

    @property
    def cost(self) -> jnp.ndarray:
        return self.__cost

    @cost.setter
    def cost(self, value: ArrayOrFloat):
        self.update_cost(value)

    def update_cost(self, cost: ArrayOrFloat):
        cost = cost if isinstance(cost, jnp.ndarray) else jnp.array(cost)
        if cost.ndim > 2:
            raise ValueError(f"the cost.ndim is too high: expected <= 2, got {cost.ndim}")
        self.__cost = cost

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
                if len(self.cost) != self.dim:
                    raise ValueError(
                        f"fail to construct matrix jnp.diag(({self.dim},)) from vector of length {self.cost.shape}"
                    )
                return jnp.diag(self.cost)
            case 2:
                if self.cost.shape != (
                    self.dim,
                    self.dim,
                ):
                    raise ValueError(f"wrong shape of the cost: {self.cost.shape} != ({self.dim}, {self.dim},)")
                return self.cost
            case _:  # pragma: no cover
                raise ValueError("fail to construct matrix cost from cost with ndim > 2")
