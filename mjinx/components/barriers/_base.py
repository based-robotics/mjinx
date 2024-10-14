from collections.abc import Callable  # noqa: F401
from typing import Generic, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components import Component, JaxComponent


@jdc.pytree_dataclass
class JaxBarrier(JaxComponent):
    """
    A base class for implementing barrier functions in JAX.

    This class provides a framework for creating barrier functions that can be used
    in optimization problems, particularly for constraint handling in robotics applications.

    Note: barrier is a function h(x) >= 0.

    :param safe_displacement_gain: The gain for computing safe displacements.
    """

    safe_displacement_gain: float

    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the barrier function value.

        :param data: The MuJoCo simulation data.
        :return: The computed barrier value.
        """
        return self.__call__(data)

    def compute_safe_displacement(self, data: mjx.Data) -> jnp.ndarray:
        r""""""
        return jnp.zeros(self.model.nv)


AtomicBarrierType = TypeVar("AtomicBarrierType", bound=JaxBarrier)


class Barrier(Generic[AtomicBarrierType], Component[AtomicBarrierType]):
    """
    A generic barrier class that wraps atomic barrier implementations.

    This class provides a high-level interface for barrier functions, allowing
    for easy integration into optimization problems.

    :param safe_displacement_gain: The gain for computing safe displacements.
    """

    safe_displacement_gain: float

    def __init__(self, name, gain, gain_fn=None, safe_displacement_gain=0, mask=None):
        """
        Initialize the Barrier object.

        :param name: The name of the barrier.
        :param gain: The gain for the barrier function.
        :param gain_fn: A function to compute the gain dynamically.
        :param safe_displacement_gain: The gain for computing safe displacements.
        :param mask: A sequence of integers to mask certain dimensions.
        """
        super().__init__(name, gain, gain_fn, mask)
        self.safe_displacement_gain = safe_displacement_gain
