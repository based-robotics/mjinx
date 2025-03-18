"""Frame task implementation."""

from collections.abc import Callable, Sequence
from typing import final

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
from jaxlie import SE3, SO3

from mjinx.components.tasks._obj_task import JaxObjTask, ObjTask
from mjinx.configuration import get_frame_jacobian_local
from mjinx.typing import ArrayOrFloat, ndarray


@jdc.pytree_dataclass
class JaxFrameTask(JaxObjTask):
    r"""
    A JAX-based implementation of a frame task for inverse kinematics.

    This class represents a task that aims to achieve a specific target frame
    for a given object in the robot model.

    The task function maps joint positions to the object's frame (pose) in the world frame:

    .. math::

        f(q) = T(q) \in SE(3)

    where :math:`T(q)` is the transformation matrix representing the object's pose.

    The error is computed using the logarithmic map in SE(3), which represents the
    relative transformation between current and target frames as a twist vector:

    .. math::

        e(q) = \log(T(q)^{-1} T_{target})

    This formulation provides a natural way to interpolate between frames and
    control both position and orientation simultaneously.

    :param target_frame: The target frame to be achieved.
    """

    target_frame: SE3

    def __post_init__(self):
        jdc._copy_and_mutate._mark_mutable(self, jdc._copy_and_mutate._Mutability.MUTABLE_NO_VALIDATION, visited=set())

        def transform_log(tau, T_bt):
            return (T_bt.multiply(SE3.exp(tau))).log()

        self._frame_jac_fn = jax.jacobian(transform_log, argnums=0)
        jdc._copy_and_mutate._mark_mutable(self, jdc._copy_and_mutate._Mutability.FROZEN, visited=set())

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        r"""
        Compute the error between the current frame and the target frame.

        The error is given by the logarithmic map in SE(3):

        .. math::

            e(q) = \log(T(q)^{-1} T_{target})

        This creates a 6D twist vector representing the relative transformation
        between the current and target frames.

        :param data: The MuJoCo simulation data.
        :return: The error vector representing the difference between the current and target frames.
        """
        return (self.get_frame(data).inverse() @ self.target_frame).log()[self.mask_idxs,]

    @final
    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        r"""
        Compute the Jacobian of the frame task.

        The Jacobian relates changes in joint positions to changes in the frame task error.
        It is computed as:

        .. math::

            J = -J_{\log} \cdot J_{frame}^T

        where:
            - :math:`J_{\log}` is the Jacobian of the logarithmic map at the relative transformation
            - :math:`J_{frame}` is the geometric Jacobian of the object's frame

        :param data: The MuJoCo simulation data.
        :return: The Jacobian matrix of the frame task.
        """
        T_bt = self.target_frame.inverse() @ self.get_frame(data).inverse()

        frame_jac = get_frame_jacobian_local(self.model, data, self.obj_id, self.obj_type)
        jlog = self._frame_jac_fn(jnp.zeros(SE3.tangent_dim), T_bt)
        return (-jlog @ frame_jac.T)[self.mask_idxs,]


class FrameTask(ObjTask[JaxFrameTask]):
    """
    A high-level representation of a frame task for inverse kinematics.

    This class provides an interface for creating and manipulating frame tasks,
    which aim to achieve a specific target frame for a object (body, geom, or site) in the robot model.

    :param name: The name of the task.
    :param cost: The cost associated with the task.
    :param gain: The gain for the task.
    :param obj_name: The name of the object to which the task is applied.
    :param gain_fn: A function to compute the gain dynamically.
    :param lm_damping: The Levenberg-Marquardt damping factor.
    :param mask: A sequence of integers to mask certain dimensions of the task.
    """

    JaxComponentType: type = JaxFrameTask
    _target_frame: SE3

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        obj_name: str,
        obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, cost, gain, obj_name, obj_type, gain_fn, lm_damping, mask)
        self.target_frame = SE3.identity()
        self._dim = SE3.tangent_dim if mask is None else len(self.mask_idxs)
        self._jax_component = jdc.replace(self._jax_component, dim=self._dim)

    @property
    def target_frame(self) -> SE3:
        """
        Get the current target frame for the task.

        :return: The current target frame as an SE3 object.
        """
        return self._target_frame

    @target_frame.setter
    def target_frame(self, value: SE3 | Sequence | ndarray):
        """
        Set the target frame for the task.

        :param value: The new target frame, either as an SE3 object or a sequence of values.
        """
        self.update_target_frame(value)
        self._jax_component = jdc.replace(self._jax_component, target_frame=self._target_frame)

    def update_target_frame(self, target_frame: SE3 | Sequence | ndarray):
        """
        Update the target frame for the task.

        This method allows setting the target frame using either an SE3 object
        or a sequence of values representing the frame.

        :param target_frame: The new target frame, either as an SE3 object or a sequence of values.
        :raises ValueError: If the provided sequence doesn't have the correct length.
        """
        if not isinstance(target_frame, SE3):
            target_frame_jnp = jnp.array(target_frame)
            if target_frame_jnp.shape[-1] != SE3.parameters_dim:
                raise ValueError(
                    "Target frame provided via array must have length 7 (xyz + quaternion with scalar first)"
                )

            xyz, quat = target_frame_jnp[..., :3], target_frame_jnp[..., 3:]
            target_frame_se3 = SE3.from_rotation_and_translation(
                SO3.from_quaternion_xyzw(
                    quat[..., [1, 2, 3, 0]],
                ),
                xyz,
            )
        else:
            target_frame_se3 = target_frame
        self._target_frame = target_frame_se3
