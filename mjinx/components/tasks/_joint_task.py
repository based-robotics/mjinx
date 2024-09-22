"""Center of mass task implementation."""

from typing import Callable, Sequence

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.configuration import get_joint_zero, joint_difference
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxJointTask(JaxTask):
    target_q: jnp.ndarray

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        r"""..."""
        return joint_difference(self.model, data.qpos, self.target_q)[self.mask_idxs,]

    # TODO: jacobian of joint task


class JointTask(Task[JaxJointTask]):
    JaxComponentType: type = JaxJointTask
    __target_q: jnp.ndarray | None

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, cost, gain, gain_fn, lm_damping, mask)
        self.__target_q = None

    def update_model(self, model: mjx.Model):
        super().update_model(model)

        self._dim = model.nq
        if len(self.mask) != self._dim:
            raise ValueError("provided mask in invalid for the model")
        if len(self.mask_idxs) != self._dim:
            self._dim = len(self.mask_idxs)

        # Validate current target_q, if empty -- set via default value
        if self.__target_q is None:
            self.target_q = get_joint_zero(model)[self.mask_idxs,]
        elif self.target_q.shape[-1] != self._dim:
            raise ValueError(
                "provided model is incompatible with target q: "
                f"{len(self.target_q)} is set, model expects {self._dim}."
            )

    @property
    def target_q(self) -> jnp.ndarray:
        if self.__target_q is None:
            raise ValueError("target value was neither provided, nor deduced from other arguments (model is missing)")
        return self.__target_q

    @target_q.setter
    def target_q(self, value: Sequence):
        self.update_target_q(value)

    def update_target_q(self, target_q: Sequence):
        target_q_jnp = jnp.array(target_q)
        if self._dim != -1 and target_q_jnp.shape[-1] != self._dim:
            raise ValueError(
                f"dimension mismatch: expected last dimension to be {self._dim}, got{target_q_jnp.shape[-1]}"
            )
        self.__target_q = target_q_jnp
