"""Frame task implementation."""

from typing import Callable, Generic, Sequence, TypeVar

import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxBodyTask(JaxTask):
    r""""""

    body_id: jdc.Static[int]


AtomicBodyTaskType = TypeVar("AtomicBodyTaskType", bound=JaxBodyTask)


class BodyTask(Generic[AtomicBodyTaskType], Task[AtomicBodyTaskType]):
    JaxComponentType: type = JaxBodyTask
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
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, cost, gain, gain_fn, lm_damping, mask)
        self.__body_name = body_name

    @property
    def body_name(self) -> str:
        return self.__body_name

    @property
    def body_id(self) -> int:
        return self.__body_id

    def update_model(self, model: mjx.Model):
        self.__body_id = mjx.name2id(
            model,
            mj.mjtObj.mjOBJ_BODY,
            self.__body_name,
        )
        if self.__body_id == -1:
            raise ValueError(f"body with name {self.__body_name} is not found.")

        return super().update_model(model)
