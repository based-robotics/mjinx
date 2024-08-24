"""Frame task implementation."""

from typing import Callable, final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import numpy as np
from typing_extensions import override

from mjinx.components.tasks.body_tasks.body_task import BodyTask, JaxBodyTask


@jdc.pytree_dataclass
class JaxPositionTask(JaxBodyTask):
    target_pos: jnp.ndarray
    axes: jdc.Static[tuple[int, ...]]

    @override
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        r""""""
        return data.xpos[self.body_id, self.axes] - self.target_pos


class PositionTask(BodyTask[JaxPositionTask]):
    __target_pos: jnp.ndarray
    __task_axes_str: str
    __task_axes_idx: tuple[int, ...]

    def __init__(
        self,
        model: mjx.Model,
        gain: np.ndarray | jnp.Array | float,
        frame_name: str,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        axes: str = "xyz",
    ):
        super().__init__(model, gain, frame_name, gain_fn, lm_damping)
        # TODO: should I create interface for that?
        self.__task_axes_str = axes
        self.__task_axes_idx = tuple([i for i in range(3) if "xyz"[i] in self.axes])

    @property
    def task_axes(self) -> str:
        return self.__task_axes_str

    @property
    def target_pos(self) -> jnp.ndarray:
        return self.__target_pos

    @target_pos.setter
    def target_pos(self, value: jnp.ndarray | np.ndarray):
        self.update_target_pos(value)

    def update_target_pos(self, target_pos: jnp.ndarray | np.ndarray):
        if len(target_pos) != len(self.__task_axes_idx):
            raise ValueError(
                "invalid dimension of the target positin value: "
                f"{len(target_pos)} given, expected {len(self.__task_axes_idx)} "
            )
        self._modified = True
        self.__target_pos = target_pos if isinstance(target_pos, jnp.ndarray) else jnp.ndarray(target_pos)

    @final
    @override
    def _build_component(self) -> JaxPositionTask:
        return JaxPositionTask(
            dim=len(self.task_axes),
            model=self.model,
            gain_function=self.gain_fn,
            lm_damping=self.lm_damping,
            target_pos=self.target_pos,
            axes=self.__task_axes_idx,
        )
