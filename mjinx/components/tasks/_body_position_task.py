"""Frame task implementation."""

from typing import Callable, Sequence, final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components.tasks._body_task import BodyTask, JaxBodyTask
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxPositionTask(JaxBodyTask):
    target_pos: jnp.ndarray

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        r""""""
        return data.xpos[self.body_id, self.mask_idxs] - self.target_pos


class PositionTask(BodyTask[JaxPositionTask]):
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
        return self.__target_pos

    @target_pos.setter
    def target_pos(self, value: Sequence):
        self.update_target_pos(value)

    def update_target_pos(self, target_pos: Sequence):
        target_pos = jnp.array(target_pos)
        if target_pos.shape[-1] != self._dim:
            raise ValueError(
                "invalid dimension of the target positin value: " f"{len(target_pos)} given, expected {self._dim} "
            )
        self.__target_pos = target_pos
