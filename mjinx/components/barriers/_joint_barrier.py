from collections.abc import Callable, Sequence

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components.barriers._base import Barrier, JaxBarrier
from mjinx.configuration import get_joint_zero, joint_difference
from mjinx.typing import ArrayOrFloat, ndarray


@jdc.pytree_dataclass
class JaxJointBarrier(JaxBarrier):
    """
    A JAX implementation of a joint barrier function.

    This class extends the JaxBarrier to specifically handle joint limits in a robotic system.
    It computes barrier values based on the current joint positions relative to their
    minimum and maximum limits.

    :param full_q_min: The minimum joint limits for all joints in the system.
    :param full_q_max: The maximum joint limits for all joints in the system.
    :param floating_base: A boolean indicating whether the robot has a floating base.
    """

    full_q_min: jnp.ndarray
    full_q_max: jnp.ndarray
    floating_base: jdc.Static[bool]

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the joint barrier values.

        This method calculates the distances between the current joint positions and
        their respective limits, considering only the joints specified by the mask.

        :param data: The MuJoCo simulation data.
        :return: An array of barrier values for the lower and upper joint limits.
        """
        mask_idxs = tuple(idx + 6 for idx in self.mask_idxs) if self.floating_base else self.mask_idxs
        return jnp.concatenate(
            [
                joint_difference(self.model, data.qpos, self.full_q_min)[mask_idxs,],
                joint_difference(self.model, self.full_q_max, data.qpos)[mask_idxs,],
            ]
        )

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the Jacobian of the joint barrier function.

        This method calculates the Jacobian matrix of the barrier function with respect
        to the joint positions, considering the mask and whether the system has a floating base.

        :param data: The MuJoCo simulation data.
        :return: The Jacobian matrix of the barrier function.
        """
        half_jac_matrix = (
            jnp.eye(self.dim // 2, self.model.nv, 6)[self.mask_idxs,]
            if self.floating_base
            else jnp.eye(self.model.nv)[self.mask_idxs,]
        )
        return jnp.vstack([half_jac_matrix, -half_jac_matrix])


class JointBarrier(Barrier[JaxJointBarrier]):
    """
    A high-level joint barrier class that wraps the JaxJointBarrier implementation.

    This class provides an interface for creating and managing joint barriers,
    including methods for updating joint limits and handling model-specific details.

    :param name: The name of the barrier.
    :param gain: The gain for the barrier function.
    :param gain_fn: A function to compute the gain dynamically.
    :param safe_displacement_gain: The gain for computing safe displacements.
    :param q_min: The minimum joint limits.
    :param q_max: The maximum joint limits.
    :param mask: A sequence of integers to mask certain joints.
    :param floating_base: A boolean indicating whether the robot has a floating base.
    """

    JaxComponentType: type = JaxJointBarrier
    _q_min: jnp.ndarray | None
    _q_max: jnp.ndarray | None

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        q_min: Sequence | None = None,
        q_max: Sequence | None = None,
        mask: Sequence[int] | None = None,
        floating_base: bool = False,
    ):
        super().__init__(name, gain, gain_fn, safe_displacement_gain, mask=mask)
        self._q_min = jnp.array(q_min) if q_min is not None else None
        self._q_max = jnp.array(q_max) if q_max is not None else None
        self.__floating_base = floating_base

    @property
    def q_min(self) -> jnp.ndarray:
        """
        Get the minimum joint limits.

        :return: The minimum joint limits.
        :raises ValueError: If q_min is not defined.
        """
        if self._q_min is None:
            raise ValueError(
                "q_min is not yet defined. Either provide it explicitly, or provide an instance of a model"
            )
        return self._q_min

    @q_min.setter
    def q_min(self, value: ndarray):
        """
        Set the minimum joint limits.

        :param value: The new minimum joint limits.
        """
        self.update_q_min(value)

    def update_q_min(self, q_min: ndarray):
        """
        Update the minimum joint limits.

        :param q_min: The new minimum joint limits.
        :raises ValueError: If the dimension of q_min is incorrect.
        """
        if q_min.shape[-1] != self.dim // 2:
            raise ValueError(
                f"[JointBarrier] wrong dimension of q_min: expected {self.dim // 2}, got {q_min.shape[-1]}"
            )

        # self.__q_min = get_joint_zero(self.model).at[self.mask_idxs_jnt_space,].set(jnp.array(q_min))
        self._q_min = jnp.array(q_min)

    @property
    def q_max(self) -> jnp.ndarray:
        """
        Get the maximum joint limits.

        :return: The maximum joint limits.
        :raises ValueError: If q_max is not defined.
        """
        if self._q_max is None:
            raise ValueError(
                "q_max is not yet defined. Either provide it explicitly, or provide an instance of a model"
            )
        return self._q_max

    @q_max.setter
    def q_max(self, value: ndarray):
        """
        Set the maximum joint limits.

        :param value: The new maximum joint limits.
        """
        self.update_q_max(value)

    def update_q_max(self, q_max: ndarray):
        """
        Update the maximum joint limits.

        :param q_max: The new maximum joint limits.
        :raises ValueError: If the dimension of q_max is incorrect.
        """

        if q_max.shape[-1] != self.dim // 2:
            raise ValueError(
                f"[JointBarrier] wrong dimension of q_max: expected {self.dim // 2}, got {q_max.shape[-1]}"
            )
        self._q_max = jnp.array(q_max)

    @property
    def mask_idxs_jnt_space(self) -> tuple[int, ...]:
        """
        Get the masked joint indices in joint space.

        :return: A tuple of masked joint indices.
        """
        if self.floating_base:
            return tuple(mask_idx + 7 for mask_idx in self.mask_idxs)
        return self.mask_idxs

    def update_model(self, model: mjx.Model):
        """
        Update the barrier with a new model.

        This method updates internal parameters based on the new model,
        including dimensions and joint limits if not previously set.

        :param model: The new MuJoCo model.
        """
        super().update_model(model)

        self._dim = 2 * self.model.nv if not self.floating_base else 2 * (self.model.nv - 6)
        self._mask = jnp.zeros(self._dim // 2)
        self._mask_idxs = tuple(range(self._dim // 2))

        begin_idx = 0 if not self.floating_base else 1
        if self._q_min is None:
            self.q_min = self.model.jnt_range[begin_idx:, 0][self.mask_idxs,]
        if self._q_max is None:
            self.q_max = self.model.jnt_range[begin_idx:, 1][self.mask_idxs,]

    @property
    def full_q_min(self) -> jnp.ndarray:
        """
        Get the full minimum joint limits vector.

        :return: The full minimum joint limits vector.
        :raises ValueError: If the model is not defined.
        """
        if self._model is None:
            raise ValueError("model is not defined yet.")
        return get_joint_zero(self.model).at[self.mask_idxs_jnt_space,].set(jnp.array(self._q_min))

    @property
    def full_q_max(self) -> jnp.ndarray:
        """
        Get the full maximum joint limits vector.

        :return: The full maximum joint limits vector.
        :raises ValueError: If the model is not defined.
        """
        if self._model is None:
            raise ValueError("model is not defined yet.")
        return get_joint_zero(self.model).at[self.mask_idxs_jnt_space,].set(jnp.array(self._q_max))

    @property
    def floating_base(self) -> bool:
        """
        Check if the robot has a floating base.

        :return: True if the robot has a floating base, False otherwise.
        """
        return self.__floating_base
