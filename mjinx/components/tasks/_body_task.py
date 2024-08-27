"""Frame task implementation."""

from typing import Callable, override

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxBodyTask(JaxTask):
    r""""""

    body_id: jdc.Static[int]


class BodyTask[T: JaxBodyTask](Task[T]):
    __body_name: str
    __body_id: int

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        body_name: str,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
    ):
        super().__init__(name, cost, gain, gain_fn, lm_damping)
        self.__body_name = body_name

    @property
    def body_name(self) -> str:
        return self.__body_name

    @property
    def body_id(self) -> int:
        return self.__body_id

    @override
    def update_model(self, model: mjx.Model):
        self.__body_id = mjx.name2id(
            model,
            mj.mjtObj.mjOBJ_BODY,
            self.__body_name,
        )
        if self.__body_id == -1:
            raise ValueError(f"body with name {self.__body_name} is not found.")

        return super().update_model(model)
