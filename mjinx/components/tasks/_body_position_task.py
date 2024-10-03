"""Frame task implementation."""

from typing import Callable, Sequence

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components.tasks._body_task import BodyTask, JaxBodyTask
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxPositionTask(JaxBodyTask):
    """
    A JAX-based implementation of a position task for inverse kinematics.

    This class represents a task that aims to achieve a specific target position
    for a body in the robot model.

    :param target_pos: The target position to be achieved.
    """

    target_pos: jnp.ndarray

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the error between the current position and the target position.

        :param data: The MuJoCo simulation data.
        :return: The error vector representing the difference between the current and target positions.
        """
        return data.xpos[self.body_id, self.mask_idxs] - self.target_pos


class PositionTask(BodyTask[JaxPositionTask]):
    """
    A high-level representation of a position task for inverse kinematics.

    This class provides an interface for creating and manipulating position tasks,
    which aim to achieve a specific target position for a body in the robot model.

    :param name: The name of the task.
    :param cost: The cost associated with the task.
    :param gain: The gain for the task.
    :param body_name: The name of the body to which the task is applied.
    :param gain_fn: A function to compute the gain dynamically.
    :param lm_damping: The Levenberg-Marquardt damping factor.
    :param mask: A sequence of integers to mask certain dimensions of the task.
    """

    JaxComponentType: type = JaxPositionTask
    __target_pos: jnp.ndarray

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        body_name: str,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, cost, gain, body_name, gain_fn, lm_damping, mask)
        self._dim = 3 if mask is None else len(self.mask_idxs)
        self.__target_pos = jnp.zeros(self._dim)

    @property
    def target_pos(self) -> jnp.ndarray:
        """
        Get the current target position for the task.

        :return: The current target position as a numpy array.
        """
        return self.__target_pos

    @target_pos.setter
    def target_pos(self, value: Sequence):
        """
        Set the target position for the task.

        :param value: The new target position as a sequence of values.
        """
        self.update_target_pos(value)

    def update_target_pos(self, target_pos: Sequence):
        """
        Update the target position for the task.

        This method allows setting the target position using a sequence of values.

        :param target_pos: The new target position as a sequence of values.
        :raises ValueError: If the provided sequence doesn't have the correct length.
        """
        target_pos = jnp.array(target_pos)
        if target_pos.shape[-1] != self._dim:
            raise ValueError(
                "invalid dimension of the target positin value: " f"{len(target_pos)} given, expected {self._dim} "
            )
        self.__target_pos = target_pos
