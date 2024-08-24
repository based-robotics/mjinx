"""Frame task implementation."""

from typing import Callable

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.tasks.base import JaxTask, Task


@jdc.pytree_dataclass
class JaxBodyTask(JaxTask):
    r""""""

    body_id: jdc.Static[int]


class BodyTask[T: JaxBodyTask](Task[T]):
    __body_name: str
    __body_id: int

    def __init__(
        self,
        model: mjx.Model,
        gain: np.ndarray | jnp.Array | float,
        frame_name: str,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
    ):
        super().__init__(model, gain, gain_fn, lm_damping)
        self.__body_name = frame_name
        self.__body_id = mjx.name2id(
            model,
            mj.mjtObj.mjOBJ_BODY,
            self.__body_name,
        )
        if self.__body_id == -1:
            raise ValueError(f"body with name {self.__body_name} is not found.")

    @property
    def body_name(self) -> str:
        return self.__body_name

    @property
    def body_id(self) -> int:
        return self.__body_id
