from typing import Generic, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components import Component, JaxComponent


@jdc.pytree_dataclass
class JaxBarrier(JaxComponent):
    r"""
    A base class for implementing barrier functions in JAX.

    This class provides a framework for creating barrier functions that can be used
    in optimization problems, particularly for constraint handling in robotics applications.

    A barrier function is defined mathematically as a function:

    .. math::

        h(q) \geq 0

    where the constraint is satisfied when h(q) is non-negative. As h(q) approaches zero,
    the barrier "activates" to prevent constraint violation. In optimization problems,
    the barrier helps enforce constraints by adding a penalty term that increases rapidly
    as the system approaches constraint boundaries.

    :param safe_displacement_gain: The gain for computing safe displacements.
    """

    safe_displacement_gain: float

    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the barrier function value.

        This method calculates the value of the barrier function h(q) at the current state.
        The barrier is active when h(q) is close to zero and satisfied when h(q) > 0.

        :param data: The MuJoCo simulation data.
        :return: The computed barrier value.
        """
        return self.__call__(data)

    def compute_safe_displacement(self, data: mjx.Data) -> jnp.ndarray:
        r"""
        Compute a safe displacement to move away from constraint boundaries.

        When the barrier function value is close to zero, this method computes
        a displacement in joint space that helps move the system away from the constraint boundary:

        .. math::

            \Delta q_{safe} = \alpha \nabla h(q)

        where:
            - :math:`\alpha` is the safe_displacement_gain
            - :math:`\nabla h(q)` is the gradient of the barrier function

        :param data: The MuJoCo simulation data.
        :return: A joint displacement vector to move away from constraint boundaries.
        """
        return jnp.zeros(self.model.nv)


AtomicBarrierType = TypeVar("AtomicBarrierType", bound=JaxBarrier)


class Barrier(Generic[AtomicBarrierType], Component[AtomicBarrierType]):
    r"""
    A generic barrier class that wraps atomic barrier implementations.

    This class provides a high-level interface for barrier functions, allowing
    for easy integration into optimization problems. Barrier functions are used
    to enforce constraints by creating a potential field that pushes the system
    away from constraint boundaries.

    In optimization problems, barriers are typically integrated as inequality constraints
    or as penalty terms in the objective function:

    .. math::

        \min_{q} f(q) \quad \text{subject to} \quad h(q) \geq 0

    or as a penalty term:

    .. math::

        \min_{q} f(q) - \lambda \log(h(q))

    where :math:`\lambda` is a weight parameter.

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
