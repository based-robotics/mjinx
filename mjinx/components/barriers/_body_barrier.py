"""Frame task implementation."""

from typing import Callable, Generic, Sequence, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.barriers._base import Barrier, JaxBarrier
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxBodyBarrier(JaxBarrier):
    """
    A JAX implementation of a body-specific barrier function.

    This class extends JaxBarrier to provide barrier functions that are
    specific to a particular body in the robot model.

    :param body_id: The ID of the body to which this barrier applies.
    """

    body_id: jdc.Static[int]


AtomicBodyBarrierType = TypeVar("AtomicBodyBarrierType", bound=JaxBodyBarrier)


class BodyBarrier(Generic[AtomicBodyBarrierType], Barrier[AtomicBodyBarrierType]):
    """
    A generic body barrier class that wraps atomic body barrier implementations.

    This class provides a high-level interface for body-specific barrier functions.

    :param body_name: The name of the body to which this barrier applies.
    """

    _body_name: str
    _body_id: int

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        body_name: str,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        mask: Sequence[int] | None = None,
    ):
        """
        Initialize the BodyBarrier object.

        :param name: The name of the barrier.
        :param gain: The gain for the barrier function.
        :param body_name: The name of the body to which this barrier applies.
        :param gain_fn: A function to compute the gain dynamically.
        :param safe_displacement_gain: The gain for computing safe displacements.
        :param mask: A sequence of integers to mask certain dimensions.
        """
        super().__init__(name, gain, gain_fn, safe_displacement_gain, mask)
        self._body_name = body_name
        self._body_id = -1

    @property
    def body_name(self) -> str:
        """
        Get the name of the body to which this barrier applies.

        :return: The name of the body.
        """
        return self._body_name

    @property
    def body_id(self) -> int:
        """
        Get the ID of the body to which this barrier applies.

        :return: The ID of the body.
        :raises ValueError: If the body ID is not available.
        """

        if self._body_id == -1:
            raise ValueError("body_id is not available until model is provided.")
        return self._body_id

    def update_model(self, model: mjx.Model):
        """
        Update the model and retrieve the body ID.

        :param model: The MuJoCo model.
        :return: The updated model.
        :raises ValueError: If the body with the specified name is not found.
        """

        self._body_id = mjx.name2id(
            model,
            mj.mjtObj.mjOBJ_BODY,
            self._body_name,
        )
        if self._body_id == -1:
            raise ValueError(f"body with name {self._body_name} is not found.")

        return super().update_model(model)
