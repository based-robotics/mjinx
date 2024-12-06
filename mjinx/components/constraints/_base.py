from typing import Generic, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components import Component, JaxComponent
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxConstraint(JaxComponent):
    def compute_constraint(self, data: mjx.Data) -> jnp.ndarray:
        return self.__call__(data)


AtomicConstraintType = TypeVar("AtomicConstraintType", bound=JaxConstraint)


class Constraint(Generic[AtomicConstraintType], Component[AtomicConstraintType]):
    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        mask=None,
    ):
        super().__init__(name, gain, gain_fn=None, mask=mask)
