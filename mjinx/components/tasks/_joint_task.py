"""Center of mass task implementation."""

import warnings
from typing import Callable, final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from typing_extensions import override

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.configuration import get_joint_zero, joint_difference
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxJointTask(JaxTask):
    target_q: jnp.ndarray

    @final
    @override
    def compute_error(self, data: mjx.Data) -> jnp.ndarray:
        r"""..."""
        return joint_difference(self.model, data.qpos, self.target_q)[self.mask_idxs]

    # TODO: jacobian of joint task


class JointTask(Task[JaxJointTask]):
    __target_q: jnp.ndarray
    __joints_mask: tuple[int, ...]

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        frame_name: str,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: np.ndarray | jnp.ndarray | None = None,
    ):
        super().__init__(name, cost, gain, frame_name, gain_fn, lm_damping, mask)

    @override
    def update_model(self, model: mjx.Model):
        super().update_model(model)
        self.target_q = get_joint_zero(model)
        self.__joints_mask = self.__generate_mask(self.__include_joints, self.__include_joints)
        self._dim = len(self.__joints_mask)

    @property
    def joints_mask(self) -> np.ndarray:
        return np.ndarray(self.__joints_mask)

    @property
    def target_q(self) -> jnp.ndarray:
        return self.__target_q

    @target_q.setter
    def target_q(self, value: jnp.ndarray | np.ndarray):
        self.update_target_q(value)

    def update_target_q(self, target_q: jnp.ndarray | np.ndarray):
        if len(target_q) != len(self.__joints_mask):
            raise ValueError(
                "invalid dimension of the target joints value: "
                f"{len(target_q)} given, expected {len(self.__task_axes_idx)} "
            )
        self._modified = True
        self.__target_q = target_q if isinstance(target_q, jnp.ndarray) else jnp.array(target_q)

    @override
    def _build_component(self) -> JaxJointTask:
        return JaxJointTask(
            dim=self.dim,
            model=self.model,
            cost=self.matrix_cost,
            gain=self.vector_gain,
            gain_function=self.gain_fn,
            lm_damping=self.lm_damping,
            target_q=self.target_q,
            mask_idxs=self.mask_idxs,
        )
