"""Center of mass task implementation."""

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.configuration import get_joint_zero
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxJointTask(JaxTask):
    """
    A JAX-based implementation of a joint task for inverse kinematics.

    This class represents a task that aims to achieve specific target joint positions
    for the robot model.

    :param full_target_q: The full target joint positions vector for all joints in the system.
    """

    full_target_q: jnp.ndarray
    qmask_idxs: jnp.ndarray

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the error between the current joint positions and the target joint positions.

        :param data: The MuJoCo simulation data.
        :return: The error vector representing the difference between the current and target joint positions.
        """
        return (data.qpos - self.full_target_q)[self.qmask_idxs,]

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the Jacobian of the joint task function.

        This method calculates the Jacobian matrix of the task function with respect
        to the joint positions, considering the mask and whether the system has a floating base.

        :param data: The MuJoCo simulation data.
        :return: The Jacobian matrix of the barrier function.
        """
        constraint_matrix = jnp.zeros((self.dim // 2, self.model.nv))
        constraint_matrix = constraint_matrix.at[jnp.arange(self.dim // 2), self.mask_idxs].set(1)
        return constraint_matrix


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
    _qmask_idxs: jnp.ndarray | None

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
        super().__init__(name, cost, gain, gain_fn, lm_damping, mask=None)
        self._final_mask = mask
        self._target_q = None
        self._floating_base = floating_base

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

        self._mask = jnp.ones(self.model.nv)
        self._qmask = jnp.ones(self.model.nq)
        for jnt_id in range(self.model.njnt):
            jnt_type = self.model.jnt_type[jnt_id]
            if jnt_type == mj.mjtJoint.mjJNT_FREE or jnt_type == mj.mjtJoint.mjJNT_BALL:
                # Calculating qpos mask
                jnt_qbegin = self.model.jnt_qposadr[jnt_id]
                jnt_qpos_size = 4 if jnt_type == mj.mjtJoint.mjJNT_BALL else 7
                jnt_qend = jnt_qbegin + jnt_qpos_size
                self._qmask = self._qmask.at[jnt_qbegin:jnt_qend].set(0)

                # Calculating qvel mask
                jnt_vbegin = self.model.jnt_dofadr[jnt_id]
                jnt_dof = 3 if jnt_type == mj.mjtJoint.mjJNT_BALL else 6
                jnt_vend = jnt_vbegin + jnt_dof
                self._mask = self._mask.at[jnt_vbegin:jnt_vend].set(0)

        # Apply the mask on top of scalar joints
        if self._final_mask is not None:
            if self._mask.sum() != len(self._final_mask):
                raise ValueError(
                    "length of provided mask should be equal to"
                    f" the number of scalar joints ({self._mask.sum()}), got length {len(self._final_mask)}"
                )
            self._mask.at[self._mask].set(self._final_mask)
            self._qmask.at[self._qmask].set(self._final_mask)

        self._qmask_idxs = jnp.argwhere(self._qmask).ravel()
        self._mask_idxs = jnp.argwhere(self._mask).ravel()

        self._dim = len(self._mask_idxs)

        # Validate current target_q, if empty -- set via default value
        if self._target_q is None:
            self.target_q = jnp.zeros(self.dim)
        elif self.target_q.shape[-1] != self.dim:
            raise ValueError(
                f"provided model is incompatible with target q: {len(self.target_q)} is set, model expects {self.dim}."
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
