"""Center of mass task implementation."""

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.configuration import get_joint_zero, joint_difference
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxJointTask(JaxTask):
    """
    A JAX-based implementation of a joint task for inverse kinematics.

    This class represents a task that aims to achieve specific target joint positions
    for the robot model.

    :param full_target_q: The full target joint positions vector for all joints in the system.
    :param floating_base: A static boolean indicating whether the robot has a floating base.
    """

    full_target_q: jnp.ndarray
    floating_base: jdc.Static[bool]

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the error between the current joint positions and the target joint positions.

        :param data: The MuJoCo simulation data.
        :return: The error vector representing the difference between the current and target joint positions.
        """
        mask_idxs = tuple(idx + 6 for idx in self.mask_idxs) if self.floating_base else self.mask_idxs
        return joint_difference(self.model, data.qpos, self.full_target_q)[mask_idxs,]

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the Jacobian of the joint task function.

        This method calculates the Jacobian matrix of the task function with respect
        to the joint positions, considering the mask and whether the system has a floating base.

        :param data: The MuJoCo simulation data.
        :return: The Jacobian matrix of the barrier function.
        """
        return (
            jnp.eye(self.dim, self.model.nv, 6)[self.mask_idxs,]
            if self.floating_base
            else jnp.eye(self.model.nv)[self.mask_idxs,]
        )


class JointTask(Task[JaxJointTask]):
    """
    A high-level representation of a joint task for inverse kinematics.

    This class provides an interface for creating and manipulating joint tasks,
    which aim to achieve specific target joint positions for the robot model.

    :param name: The name of the task.
    :param cost: The cost associated with the task.
    :param gain: The gain for the task.
    :param gain_fn: A function to compute the gain dynamically.
    :param lm_damping: The Levenberg-Marquardt damping factor.
    :param mask: A sequence of integers to mask certain dimensions of the task.
    :param floating_base: A boolean indicating whether the robot has a floating base.
    """

    JaxComponentType: type = JaxJointTask
    _target_q: jnp.ndarray | None
    _floating_base: bool

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Sequence[int] | None = None,
        floating_base: bool = False,
    ):
        super().__init__(name, cost, gain, gain_fn, lm_damping, mask)
        self._target_q = None
        self._floating_base = floating_base

    @property
    def mask_idxs_jnt_space(self) -> tuple[int, ...]:
        """
        Get the masked joint indices in joint space.

        :return: A tuple of masked joint indices, adjusted for floating base if applicable.
        """
        if self.floating_base:
            return tuple(mask_idx + 7 for mask_idx in self.mask_idxs)
        return self.mask_idxs

    def update_model(self, model: mjx.Model):
        """
        Update the MuJoCo model and set the joint dimensions for the task.

        This method is called when the model is updated or when the task
        is first added to the problem. It adjusts the dimensions based on
        whether the robot has a floating base and validates the mask.

        :param model: The new MuJoCo model.
        :raises ValueError: If the provided mask is invalid for the model or if the target_q is incompatible.
        """
        super().update_model(model)

        self._dim = model.nv if not self.floating_base else model.nv - 6
        # if self.floating_base:
        if len(self.mask) != self.dim:
            raise ValueError("provided mask in invalid for the model")
        if len(self.mask_idxs) != self.dim:
            self._dim = len(self.mask_idxs)

        # Validate current target_q, if empty -- set via default value
        if self._target_q is None:
            self.target_q = jnp.zeros(self.dim)
        elif self.target_q.shape[-1] != self.dim:
            raise ValueError(
                "provided model is incompatible with target q: "
                f"{len(self.target_q)} is set, model expects {self.dim}."
            )

    @property
    def target_q(self) -> jnp.ndarray:
        """
        Get the current target joint positions for the task.

        :return: The current target joint positions as a numpy array.
        :raises ValueError: If the target value was not provided and the model is missing.
        """
        if self._target_q is None:
            raise ValueError("target value was neither provided, nor deduced from other arguments (model is missing)")
        return self._target_q

    @target_q.setter
    def target_q(self, value: Sequence):
        """
        Set the target joint positions for the task.

        :param value: The new target joint positions as a sequence of values.
        """
        self.update_target_q(value)

    def update_target_q(self, target_q: Sequence):
        """
        Update the target joint positions for the task.

        This method allows setting the target joint positions using a sequence of values.

        :param target_q: The new target joint positions as a sequence of values.
        :raises ValueError: If the provided sequence doesn't have the correct length.
        """
        target_q_jnp = jnp.array(target_q)
        if self._dim != -1 and target_q_jnp.shape[-1] != self._dim:
            raise ValueError(
                f"dimension mismatch: expected last dimension to be {self._dim}, got{target_q_jnp.shape[-1]}"
            )
        self._target_q = target_q_jnp

    @property
    def full_target_q(self) -> jnp.ndarray:
        """
        Get the full target joint positions vector for all joints in the system.

        :return: The full target joint positions vector.
        :raises ValueError: If the model is not defined yet.
        """
        if self._model is None:
            raise ValueError("model is not defined yet.")
        return get_joint_zero(self.model).at[self.mask_idxs_jnt_space,].set(self._target_q)

    @property
    def floating_base(self) -> bool:
        """
        Check if the robot has a floating base.

        :return: True if the robot has a floating base, False otherwise.
        """
        return self._floating_base
