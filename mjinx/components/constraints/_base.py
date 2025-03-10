from typing import Generic, Sequence, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components import Component, JaxComponent
from mjinx.typing import ArrayLike, ArrayOrFloat


@jdc.pytree_dataclass
class JaxConstraint(JaxComponent):
    active: bool
    hard_constraint: jdc.Static[bool]
    soft_constraint_cost: jnp.ndarray

    def compute_constraint(self, data: mjx.Data) -> jnp.ndarray:
        return self.__call__(data)


AtomicConstraintType = TypeVar("AtomicConstraintType", bound=JaxConstraint)


class Constraint(Generic[AtomicConstraintType], Component[AtomicConstraintType]):
    active: bool
    _soft_constraint_cost: jnp.ndarray | None
    hard_constraint: bool

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        mask: Sequence[int] = None,
        hard_constraint: bool = False,
        soft_constraint_cost: ArrayLike | None = None,
    ):
        super().__init__(name, gain, gain_fn=None, mask=mask)
        self.active = True
        self.hard_constraint = hard_constraint
        self._soft_constraint_cost = jnp.array(soft_constraint_cost) if soft_constraint_cost is not None else None

    @property
    def soft_constraint_cost(self) -> jnp.ndarray:
        # return self._soft_constraint_cost if self._soft_constraint_cost is not None else 1e2 * jnp.eye(self.dim)
        if self._soft_constraint_cost is None:
            return 1e2 * jnp.eye(self.dim)

        match self._soft_constraint_cost.ndim:
            case 0:
                return jnp.eye(self.dim) * self._soft_constraint_cost
            case 1:
                if len(self._soft_constraint_cost) != self.dim:
                    raise ValueError(
                        f"fail to construct matrix jnp.diag(({self.dim},)) from vector of length {self._soft_constraint_cost.shape}"
                    )
                return jnp.diag(self._soft_constraint_cost)
            case 2:
                if self._soft_constraint_cost.shape != (
                    self.dim,
                    self.dim,
                ):
                    raise ValueError(
                        f"wrong shape of the cost: {self._soft_constraint_cost.shape} != ({self.dim}, {self.dim},)"
                    )
                return self._soft_constraint_cost
            case _:  # pragma: no cover
                raise ValueError("fail to construct cost, given dim > 2")
