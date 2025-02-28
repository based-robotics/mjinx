from collections.abc import Sequence
from typing import TypeVar

import jax.numpy as jnp  # noqa: F401
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.constraints._base import Constraint, JaxConstraint
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxModelEqualityConstraint(JaxConstraint):
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return data.efc_pos[self.mask_idxs]

    def compute_jacobian(self, data):
        return data.efc_J.reshape(data.nefc, self.model.nv)[self.mask_idxs]


AtomicModelEqualityConstraintType = TypeVar("AtomicModelEqualityConstraintType", bound=JaxModelEqualityConstraint)


class ModelEqualityConstraint(Constraint[AtomicModelEqualityConstraintType]):
    """Accumulates all equality constraints present in the model."""

    JaxComponentType = JaxModelEqualityConstraint
    SUPPORTED_EQ_TYPES: tuple[mj.mjtEq] = (
        mj.mjtEq.mjEQ_CONNECT,
        mj.mjtEq.mjEQ_WELD,
        mj.mjtEq.mjEQ_JOINT,
    )

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
    ):
        super().__init__(name, gain, mask=None)
        self.active = True

    def update_model(self, model):
        super().update_model(model)

        self._mask = jnp.isin(model.eq_type, jnp.array(self.SUPPORTED_EQ_TYPES, dtype=jnp.int16)).astype(jnp.uint16)
        self._mask_idxs = jnp.argwhere(self._mask).flatten()
        print(self._mask_idxs)
        self._dim = len(self.mask)
