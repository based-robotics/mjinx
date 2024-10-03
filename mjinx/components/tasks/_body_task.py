"""Frame task implementation."""

from typing import Callable, Generic, Sequence, TypeVar

import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxBodyTask(JaxTask):
    """
    A JAX-based implementation of a body task for inverse kinematics.

    This class serves as a base for tasks that are applied to specific bodies
    in the robot model.

    :param body_id: The ID of the body to which the task is applied.
    """

    body_id: jdc.Static[int]


AtomicBodyTaskType = TypeVar("AtomicBodyTaskType", bound=JaxBodyTask)


class BodyTask(Generic[AtomicBodyTaskType], Task[AtomicBodyTaskType]):
    """
    A high-level representation of a body task for inverse kinematics.

    This class provides an interface for creating and manipulating tasks
    that are applied to specific bodies in the robot model.

    :param name: The name of the task.
    :param cost: The cost associated with the task.
    :param gain: The gain for the task.
    :param body_name: The name of the body to which the task is applied.
    :param gain_fn: A function to compute the gain dynamically.
    :param lm_damping: The Levenberg-Marquardt damping factor.
    :param mask: A sequence of integers to mask certain dimensions of the task.
    """

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
        """
        Get the name of the body to which the task is applied.

        :return: The name of the body.
        """
        return self.__body_name

    @property
    def body_id(self) -> int:
        """
        Get the ID of the body to which the task is applied.

        :return: The ID of the body.
        """
        return self.__body_id

    def update_model(self, model: mjx.Model):
        """
        Update the MuJoCo model and set the body ID for the task.

        This method is called when the model is updated or when the task
        is first added to the problem.

        :param model: The new MuJoCo model.
        :raises ValueError: If the body with the specified name is not found in the model.
        """
        self.__body_id = mjx.name2id(
            model,
            mj.mjtObj.mjOBJ_BODY,
            self.__body_name,
        )
        if self.__body_id == -1:
            raise ValueError(f"body with name {self.__body_name} is not found.")

        return super().update_model(model)
