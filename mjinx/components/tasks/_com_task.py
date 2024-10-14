"""Center of mass task implementation."""

from collections.abc import Callable, Sequence
from typing import final

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco
import mujoco.mjx as mjx
from mujoco.mjx._src import scan

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.configuration import jac_dq2v
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxComTask(JaxTask):
    """
    A JAX-based implementation of a center of mass (CoM) task for inverse kinematics.

    This class represents a task that aims to achieve a specific target center of mass
    for the robot model.

    :param target_com: The target center of mass position to be achieved.
    """

    target_com: jnp.ndarray

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the error between the current center of mass and the target center of mass.

        :param data: The MuJoCo simulation data.
        :return: The error vector representing the difference between the current and target center of mass.
        """
        error = data.subtree_com[self.model.body_rootid[0], self.mask_idxs] - self.target_com
        return error

    @final
    def compute_jacobian(self, data):
        def specific_update(model: mjx.Model, q: jnp.ndarray) -> mjx.Data:
            data = mjx.kinematics(model, mjx.make_data(model).replace(qpos=q))

            # calculate center of mass of each subtree
            def subtree_sum(carry, xipos, body_mass):
                pos, mass = xipos * body_mass, body_mass
                if carry is not None:
                    subtree_pos, subtree_mass = carry
                    pos, mass = pos + subtree_pos, mass + subtree_mass
                return pos, mass

            pos, mass = scan.body_tree(model, subtree_sum, "bb", "bb", data.xipos, model.body_mass, reverse=True)
            cond = jnp.tile(mass < mujoco.mjMINVAL, (3, 1)).T
            # take maximum to avoid NaN in gradient of jp.where
            subtree_com = jax.vmap(jnp.divide)(pos, jnp.maximum(mass, mujoco.mjMINVAL))
            subtree_com = jnp.where(cond, data.xipos, subtree_com)
            data = data.replace(subtree_com=subtree_com)

            return data

        jac = jax.jacrev(
            lambda q, model=self.model: self.__call__(
                specific_update(model, q),
            ),
            argnums=0,
        )(data.qpos)
        if self.model.nq != self.model.nv:
            jac = jac @ jac_dq2v(self.model, data.qpos)
        return jac


class ComTask(Task[JaxComTask]):
    """
    A high-level representation of a center of mass (CoM) task for inverse kinematics.

    This class provides an interface for creating and manipulating center of mass tasks,
    which aim to achieve a specific target center of mass for the robot model.

    :param name: The name of the task.
    :param cost: The cost associated with the task.
    :param gain: The gain for the task.
    :param gain_fn: A function to compute the gain dynamically.
    :param lm_damping: The Levenberg-Marquardt damping factor.
    :param mask: A sequence of integers to mask certain dimensions of the task.
    """

    JaxComponentType: type = JaxComTask
    _target_com: jnp.ndarray

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Sequence[int] | None = None,
    ):
        if mask is not None and len(mask) != 3:
            raise ValueError("provided mask is too large, expected 1D vector of length 3")
        super().__init__(name, cost, gain, gain_fn, lm_damping, mask=mask)
        self._dim = 3 if mask is None else len(self.mask_idxs)
        self.target_com = jnp.zeros(self._dim)

    @property
    def target_com(self) -> jnp.ndarray:
        """
        Get the current target center of mass for the task.

        :return: The current target center of mass as a numpy array.
        """
        return self._target_com

    @target_com.setter
    def target_com(self, value: Sequence):
        """
        Set the target center of mass for the task.

        :param value: The new target center of mass as a sequence of values.
        """
        self.update_target_com(value)

    def update_target_com(self, target_com: Sequence):
        """
        Update the target center of mass for the task.

        This method allows setting the target center of mass using a sequence of values.

        :param target_com: The new target center of mass as a sequence of values.
        :raises ValueError: If the provided sequence doesn't have the correct length.
        """
        target_com_jnp = jnp.array(target_com)
        if target_com_jnp.shape[-1] != self._dim:
            raise ValueError(
                "invalid last dimension of target CoM : " f"{target_com_jnp.shape[-1]} given, expected {self._dim} "
            )
        self._target_com = target_com_jnp
