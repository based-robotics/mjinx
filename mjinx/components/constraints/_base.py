from typing import Generic, Sequence, TypeVar, Callable

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components import Component, JaxComponent
from mjinx.typing import ArrayLike, ArrayOrFloat


@jdc.pytree_dataclass
class JaxConstraint(JaxComponent):
    """
    A JAX-based representation of an equality constraint.

    This class defines an equality constraint function f(x) = 0 for optimization tasks.

    :param hard_constraint: A flag that specifies if the constraint is hard (True) or soft (False).
    :param soft_constraint_cost: The cost matrix used for soft constraint relaxation.
    """

    hard_constraint: jdc.Static[bool]
    soft_constraint_cost: jnp.ndarray

    def compute_constraint(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the equality constraint function value.

        Evaluates the constraint function f(x) = 0 based on the current simulation data.
        For soft constraints, the computed value is later penalized by the associated cost;
        for hard constraints, the evaluation is directly used in the Ax = b formulation.

        :param data: The MuJoCo simulation data.
        :return: The computed constraint value.
        """
        return self.__call__(data)


AtomicConstraintType = TypeVar("AtomicConstraintType", bound=JaxConstraint)


class Constraint(Generic[AtomicConstraintType], Component[AtomicConstraintType]):
    r"""
    A high-level component for formulating equality constraints.

    This class wraps an atomic JAX constraint (JaxConstraint) and provides a framework to
    manage constraints within the optimization problem. Equality constraints are specified as:

    .. math::

        f(x) = 0

    They can be treated in one of two ways:
      - As a soft constraint. In this mode, the constraint violation is penalized by a cost
        (typically with a high gain), transforming the constraint into a task.
      - As a hard constraint. Here, the constraint is enforced as a strict equality in the following form:

    .. math::

        \nabla h(q)^T v = -\alpha h(q),

    where :math:`\alpha` controls constraint enforcement and :math:`v` is the velocity vector.

    :param matrix_cost: The cost matrix associated with the task.
    :param lm_damping: The Levenberg-Marquardt damping factor.
    :param hard_constraint: Indicates whether the constraint is hard (True) or soft (False).
    """

    _soft_constraint_cost: jnp.ndarray | None
    hard_constraint: bool

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        mask: Sequence[int] | None = None,
        hard_constraint: bool = False,
        soft_constraint_cost: ArrayLike | None = None,
    ):
        """
        Initialize a Constraint object.

        Sets up the constraint with the given parameters. Depending on whether it is
        a hard or soft constraint, further integration in the optimization problem will
        process it accordingly.

        :param name: The unique identifier for the constraint.
        :param gain: The gain for the constraint function, affecting its impact.
        :param mask: Indices to select specific dimensions for evaluation. If not provided, applies to all dimensions.
        :param hard_constraint: If True, the constraint is handled as a hard constraint. Defaults to False (i.e. soft constraint).
        :param soft_constraint_cost: The cost used to relax a soft constraint. If not provided, a default high gain cost matrix (scaled identity matrix) based on the component dimension will be used.
        """
        super().__init__(name, gain, gain_fn=None, mask=mask)
        self.hard_constraint = hard_constraint
        self._soft_constraint_cost = jnp.array(soft_constraint_cost) if soft_constraint_cost is not None else None
        self._jax_component = jdc.replace(
            self._jax_component,
            hard_constraint=hard_constraint,
        )

    @property
    def soft_constraint_cost(self) -> jnp.ndarray:
        """
        Get the cost matrix associated with a soft constraint.

        For soft constraints, any violation is penalized by a cost if self._soft_constraint_cost is not None else 1e2 * jnp.eye(self.dim)"

        :return: The cost matrix for the soft constraint.
        """
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
