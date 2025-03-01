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
        return data.efc_pos[data.efc_type == mj.mjtConstraint.mjCNSTR_EQUALITY]

    def compute_jacobian(self, data):
        print(data.efc_J.shape)
        print(data.nefc, self.model.nv)
        print(data.efc_type == mj.mjtConstraint.mjCNSTR_EQUALITY)
        return data.efc_J[data.efc_type == mj.mjtConstraint.mjCNSTR_EQUALITY, :]


AtomicModelEqualityConstraintType = TypeVar("AtomicModelEqualityConstraintType", bound=JaxModelEqualityConstraint)


class ModelEqualityConstraint(Constraint[AtomicModelEqualityConstraintType]):
    """Accumulates all equality constraints present in the model."""

    JaxComponentType = JaxModelEqualityConstraint

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
    ):
        super().__init__(name, gain, mask=None)
        self.active = True

    def update_model(self, model):
        super().update_model(model)

        nefc = 0
        for i in range(self.model.neq):
            match self.model.eq_type[i]:
                case mj.mjtEq.mjEQ_CONNECT:
                    nefc += 3
                case mj.mjtEq.mjEQ_WELD:
                    nefc += 6
                case mj.mjtEq.mjEQ_JOINT:
                    nefc += 1
                case _:
                    raise ValueError(f"Unsupported equality constraint type {self.model.eq_type[i]}")

        self._mask_idxs = jnp.arange(nefc)
        self._dim = nefc
