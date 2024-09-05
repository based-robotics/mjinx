"""Frame task implementation."""

from typing import Callable, Iterable, final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.tasks._body_task import BodyTask, JaxBodyTask
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxPositionTask(JaxBodyTask):
    target_pos: jnp.ndarray

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        r""""""
        return data.xpos[self.body_id, self.mask_idxs] - self.target_pos


class PositionTask(BodyTask[JaxPositionTask]):
    __target_pos: jnp.ndarray

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
        self._dim = 3 if mask is None else len(self.mask_idxs)

    @property
    def target_pos(self) -> jnp.ndarray:
        return self.__target_pos

    @target_pos.setter
    def target_pos(self, value: jnp.ndarray | np.ndarray):
        self.update_target_pos(value)

    def update_target_pos(self, target_pos: jnp.ndarray | np.ndarray):
        if len(target_pos) != self._dim:
            raise ValueError(
                "invalid dimension of the target positin value: "
                f"{len(target_pos)} given, expected {len(self.__task_axes_idx)} "
            )
        self._modified = True
        self.__target_pos = target_pos if isinstance(target_pos, jnp.ndarray) else jnp.array(target_pos)

    @final
    def _build_component(self) -> JaxPositionTask:
        return JaxPositionTask(
            dim=self.dim,
            model=self.model,
            cost=self.matrix_cost,
            gain=self.vector_gain,
            gain_function=self.gain_fn,
            lm_damping=self.lm_damping,
            target_pos=self.target_pos,
            mask_idxs=self.mask_idxs,
        )
