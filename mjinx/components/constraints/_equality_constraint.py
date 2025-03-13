from collections.abc import Sequence
from typing import TypeVar

import jax.numpy as jnp  # noqa: F401
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.constraints._base import Constraint, JaxConstraint
import mjinx.typing


@jdc.pytree_dataclass
class JaxModelEqualityConstraint(JaxConstraint):
    """
    A JAX-based equality constraint derived from the simulation model.

    Equality constraints respresent all constrainst in the mujoco, which could be defined in
    <equality> tag in the xml file. More details could be found here: https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality.

    This class utilized the fact, that during kinematics computations, the equality value are computed as well,
    and the jacobian of the equalities are stored in data.efc_J, and the residuals are stored in data.efc_pos.

    :param data: A MuJoCo simulation data structure containing model-specific constraint information.
    :return: A jax.numpy.ndarray with the positions corresponding to the equality constraints.
    """

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the equality constraint values from the simulation data.

        This method selects the positions of equality constraints from the simulation data
        by filtering based on the constraint type and applying the mask indices.

        :param data: The MuJoCo simulation data.
        :return: A jax.numpy.ndarray containing the equality constraint values.
        """
        return data.efc_pos[data.efc_type == mj.mjtConstraint.mjCNSTR_EQUALITY][self.mask_idxs,]

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the Jacobian matrix of the equality constraint.

        Retrieves the relevant rows of the Jacobian matrix from the simulation data,
        filtering by equality constraint type and applying the mask indices.

        :param data: The MuJoCo simulation data.
        :return: A jax.numpy.ndarray representing the Jacobian matrix of the equality constraints.
        """
        return data.efc_J[data.efc_type == mj.mjtConstraint.mjCNSTR_EQUALITY, :][self.mask_idxs, :]


AtomicModelEqualityConstraintType = TypeVar("AtomicModelEqualityConstraintType", bound=JaxModelEqualityConstraint)


class ModelEqualityConstraint(Constraint[AtomicModelEqualityConstraintType]):
    """
    High-level component that aggregates all equality constraints from the simulation model.

    The main purpose of the wrapper is to recalculate mask based on the dimensions of the equality constrain
    and compute proper dimensionality.


    :param name: The unique identifier for the constraint.
    :param gain: The gain for the constraint function, affecting its impact.
    :param hard_constraint: If True, the constraint is handled as a hard constraint.
                            Defaults to False (i.e., soft constraint).
    :param soft_constraint_cost: The cost used to relax a soft constraint. If not provided, a default high gain
                                 cost matrix (scaled identity matrix) based on the component dimension will be used.
    """

    JaxComponentType = JaxModelEqualityConstraint

    def __init__(
        self,
        name: str = "equality_constraint",
        gain: mjinx.typing.ArrayOrFloat = 100,
        hard_constraint: bool = False,
        soft_constraint_cost: mjinx.typing.ArrayLike | None = None,
    ):
        """
        Initialize a new equality constraint.

        Args:
            name (str): A unique identifier for the constraint. Defaults to 'equality_constraint'.
            gain (ArrayOrFloat): The gain used to weight or scale the constraint. Can be a float or an array-like type. Defaults to 100.
            hard_constraint (bool): Flag to determine if the constraint should be enforced strictly (hard constraint). Defaults to False.
            soft_constraint_cost (ArrayLike | None): Optional cost associated with treating the constraint as a soft constraint.
                When provided, this cost is used to penalize deviations.

        Attributes:
            active (bool): Indicates whether the constraint is active. Initialized to True.
        """
        super().__init__(
            name=name,
            gain=gain,
            mask=None,
            hard_constraint=hard_constraint,
            soft_constraint_cost=soft_constraint_cost,
        )
        self.active = True

    def update_model(self, model: mj.MjModel) -> None:
        """
        Update the equality constraint using the provided MuJoCo model.

        This method computes the total dimension of the equality constraints by iterating
        over the equality types in the model. Based on the type of each equality constraint,
        it sets the mask indices and component dimension accordingly.

        :param model: The MuJoCo model instance.
        :return: None
        """
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
