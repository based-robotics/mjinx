"""Center of mass task implementation."""

from typing import Callable, Iterable, final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.configuration import get_joint_zero, joint_difference
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxJointTask(JaxTask):
    target_q: jnp.ndarray

    @final
    def compute_error(self, data: mjx.Data) -> jnp.ndarray:
        r"""..."""
        return joint_difference(self.model, data.qpos, self.target_q)[self.mask_idxs]

    # TODO: jacobian of joint task


class JointTask(Task[JaxJointTask]):
    __target_q: jnp.ndarray | None
    __joints_mask: tuple[int, ...]

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        frame_name: str,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Iterable | None = None,
    ):
        super().__init__(name, cost, gain, frame_name, gain_fn, lm_damping, mask)
        self.__target_q = None

    def update_model(self, model: mjx.Model):
        super().update_model(model)
        self._dim = len(self.__joints_mask) if len(self.mask_idxs) == 0 else len(self.mask_idxs)
        if self.target_q is None:
            self.target_q = get_joint_zero(model)
        elif len(self.target_q) != self._dim:
            raise ValueError(
                "provided model is incompatible with target q: "
                f"{len(self.target_q)} is set, model expects {len(self._dim)}."
            )
        self.__joints_mask = self.__generate_mask(self.__include_joints, self.__include_joints)

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
        self._modified = True
        self.__target_q = target_q if isinstance(target_q, jnp.ndarray) else jnp.array(target_q)

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
