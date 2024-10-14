from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components._base import Component, JaxComponent
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxTask(JaxComponent):
    """
    A JAX-based implementation of a task for inverse kinematics.

    This class serves as a base for all tasks in the inverse kinematics problem.

    :param matrix_cost: The cost matrix associated with the task.
    :param lm_damping: The Levenberg-Marquardt damping factor.
    """

    matrix_cost: jnp.ndarray
    lm_damping: jdc.Static[float]

    def compute_error(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the error for the task.

        This method is equivalent to calling the task object directly.

        :param data: The MuJoCo simulation data.
        :return: The error vector for the task.
        """
        return self.__call__(data)


AtomicTaskType = TypeVar("AtomicTaskType", bound=JaxTask)


class Task(Generic[AtomicTaskType], Component[AtomicTaskType]):
    """
    A high-level representation of a task for inverse kinematics.

    This class provides an interface for creating and manipulating tasks
    in the inverse kinematics problem.

    :param name: The name of the task.
    :param cost: The cost associated with the task.
    :param gain: The gain for the task.
    :param gain_fn: A function to compute the gain dynamically.
    :param lm_damping: The Levenberg-Marquardt damping factor.
    :param mask: A sequence of integers to mask certain dimensions of the task.
    """

    lm_damping: float
    _cost: jnp.ndarray

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Sequence[int] | None = None,
    ):
        """
        Initialize a Task object.

        :param name: The name of the task.
        :param cost: The cost associated with the task. Can be a scalar, vector, or matrix.
        :param gain: The gain for the task. Can be a scalar or vector.
        :param gain_fn: A function to compute the gain dynamically. If None, a default function is used.
        :param lm_damping: The Levenberg-Marquardt damping factor. Must be non-negative.
        :param mask: A sequence of integers to mask certain dimensions of the task. If None, all dimensions are used.
        :raises ValueError: If lm_damping is negative.
        """
        super().__init__(name, gain, gain_fn, mask)
        if lm_damping < 0:
            raise ValueError("lm_damping has to be positive")
        self.lm_damping = lm_damping

        self.update_cost(cost)

    @property
    def cost(self) -> jnp.ndarray:
        """
        Get the cost associated with the task.

        :return: The cost as a numpy array.
        """
        return self._cost

    @cost.setter
    def cost(self, value: ArrayOrFloat):
        """
        Set the cost for the task.

        :param value: The new cost value.
        """
        self.update_cost(value)

    def update_cost(self, cost: ArrayOrFloat):
        """
        Update the cost for the task.

        This method allows setting the cost using either a scalar, vector, or matrix.

        :param cost: The new cost value.
        :raises ValueError: If the cost has an invalid dimension.
        """
        cost_jnp = cost if isinstance(cost, jnp.ndarray) else jnp.array(cost)
        if cost_jnp.ndim > 2:
            raise ValueError(f"the cost.ndim is too high: expected <= 2, got {cost_jnp.ndim}")
        self._cost = cost_jnp

    @property
    def matrix_cost(self) -> jnp.ndarray:
        """
        Get the cost matrix associated with the task.

        This method converts cost to matrix form as follows:
            - cost is scalar -> matrix_cost = jnp.eye(self.dim) * cost
            - cost is vector -> matrix_cost = jnp.diag(cost)
            - cost is matrix -> matrix_cost = cost

        :return: The cost matrix as a numpy array.
        :raises ValueError: If the dimension is not set or the cost size is invalid.
        """

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
