from collections.abc import Callable, Sequence

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.barriers._base import Barrier, JaxBarrier
from mjinx.typing import ArrayOrFloat, ndarray


@jdc.pytree_dataclass
class JaxJointBarrier(JaxBarrier):
    r"""
    A JAX implementation of a joint barrier function.

    This class extends the JaxBarrier to specifically handle joint limits in a robotic system.
    It computes barrier values based on the current joint positions relative to their
    minimum and maximum limits.
    
    The joint barrier enforces that joint positions remain within their specified limits:

    .. math::

        h_{min}(q) &= q - q_{min} \geq 0 \\
        h_{max}(q) &= q_{max} - q \geq 0

    where:
        - :math:`q` is the joint position vector
        - :math:`q_{min}` is the vector of minimum joint limits
        - :math:`q_{max}` is the vector of maximum joint limits

    :param q_min: The minimum joint limits.
    :param q_max: The maximum joint limits.
    """

    q_min: jnp.ndarray
    q_max: jnp.ndarray
    qmask_idxs: jnp.ndarray

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        r"""
        Compute the joint barrier values.

        This method calculates two sets of barrier values:
        1. Lower limit barriers: :math:`q - q_{min} \geq 0`
        2. Upper limit barriers: :math:`q_{max} - q \geq 0`

        These barriers ensure that joint positions remain within their allowable ranges.

        :param data: The MuJoCo simulation data.
        :return: An array of barrier values for the lower and upper joint limits.
        """
        return jnp.concatenate(
            [
                data.qpos[self.qmask_idxs,] - self.q_min,
                self.q_max - data.qpos[self.qmask_idxs],
            ]
        )

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        r"""
        Compute the Jacobian of the joint barrier function.

        For joint barriers, the Jacobian has a simple form:
        - For lower limits (:math:`q - q_{min}`): :math:`J = I` (identity matrix)
        - For upper limits (:math:`q_{max} - q`): :math:`J = -I` (negative identity matrix)

        These are combined into a single Jacobian matrix, with appropriate
        adjustments for masked joints and floating base.

        :param data: The MuJoCo simulation data.
        :return: The Jacobian matrix of the barrier function.
        """
        constraint_matrix = jnp.zeros((self.dim // 2, self.model.nv))
        constraint_matrix = constraint_matrix.at[jnp.arange(self.dim // 2), self.mask_idxs].set(1)
        return jnp.vstack([constraint_matrix, -constraint_matrix])


class JointBarrier(Barrier[JaxJointBarrier]):
    """
    A high-level joint barrier class that wraps the JaxJointBarrier implementation.

    This class provides an interface for creating and managing joint barriers,
    including methods for updating joint limits and handling model-specific details.

    Joint barriers are one of the most common constraint types in robotics, ensuring that
    joint positions respect their mechanical limits. They can be incorporated into
    optimization problems to prevent solutions that would require impossible joint configurations.

    The barrier function is formulated to create a potential field that increases as
    joints approach their limits, effectively pushing the optimization away from invalid regions.

    :param name: The name of the barrier.
    :param gain: The gain for the barrier function.
    :param gain_fn: A function to compute the gain dynamically.
    :param safe_displacement_gain: The gain for computing safe displacements.
    :param q_min: The minimum joint limits.
    :param q_max: The maximum joint limits.
    :param mask: A sequence of integers to mask certain joints.
    """

    JaxComponentType: type = JaxJointBarrier
    _q_min: jnp.ndarray | None
    _q_max: jnp.ndarray | None
    _qmask: jnp.ndarray | None
    _qmask_idxs: jnp.ndarray | None

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        q_min: Sequence | None = None,
        q_max: Sequence | None = None,
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, gain, gain_fn, safe_displacement_gain, mask=None)
        self._q_min = jnp.array(q_min) if q_min is not None else None
        self._q_max = jnp.array(q_max) if q_max is not None else None
        self._final_mask = mask

    @property
    def q_min(self) -> jnp.ndarray:
        r"""
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

    def update_model(self, model: mjx.Model):
        """
        Update the barrier with a new model.

        This method updates internal parameters based on the new model,
        including dimensions and joint limits if not previously set.

        :param model: The new MuJoCo model.
        """
        super().update_model(model)

        self._mask = jnp.ones(self.model.nv, dtype=jnp.uint32)
        self._qmask = jnp.ones(self.model.nq, dtype=jnp.uint32)
        jnt_mask = (self.model.jnt_type != mj.mjtJoint.mjJNT_FREE) & (self.model.jnt_type != mj.mjtJoint.mjJNT_BALL)
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
            self._mask = self._mask.at[self._mask.astype(jnp.bool)].set(self._final_mask)
            self._qmask = self._qmask.at[self._qmask.astype(jnp.bool)].set(self._final_mask)

        self._qmask_idxs = jnp.argwhere(self._qmask).ravel()
        self._mask_idxs = jnp.argwhere(self._mask).ravel()

        self._dim = 2 * len(self._mask_idxs)

        if self._q_min is None:
            self.q_min = self.model.jnt_range[:, 0][jnt_mask]
        if self._q_max is None:
            self.q_max = self.model.jnt_range[:, 1][jnt_mask]

    @property
    def qmask_idxs(self) -> jnp.ndarray:
        """
        Get the indices of the masked joints.

        :return: The indices of the masked joints.
        """
        return self._qmask_idxs
