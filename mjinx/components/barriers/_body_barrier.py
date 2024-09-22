"""Frame task implementation."""

from typing import Callable, Generic, Sequence, TypeVar

import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.barriers._base import Barrier, JaxBarrier
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxBodyBarrier(JaxBarrier):
    r""""""

    body_id: jdc.Static[int]


AtomicBodyBarrierType = TypeVar("AtomicBodyBarrierType", bound=JaxBodyBarrier)


class BodyBarrier(Generic[AtomicBodyBarrierType], Barrier[AtomicBodyBarrierType]):
    __body_name: str
    __body_id: int

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        body_name: str,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, gain, gain_fn, safe_displacement_gain, mask)
        self.__body_name = body_name
        self.__body_id = -1

    @property
    def body_name(self) -> str:
        return self.__body_name

    @property
    def body_id(self) -> int:
        if self.__body_id == -1:
            raise ValueError("body_id is not available until model is provided.")
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
