from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.barriers._obj_barrier import JaxObjBarrier, ObjBarrier
from mjinx.typing import ArrayOrFloat, PositionLimitType


@jdc.pytree_dataclass
class JaxPositionBarrier(JaxObjBarrier):
    """
    A JAX implementation of a position barrier function for a specific object (body, geometry, or site).

    This class extends JaxObjBarrier to provide position-specific barrier functions.

    :param p_min: The minimum allowed position.
    :param p_max: The maximum allowed position.
    """

    p_min: jnp.ndarray
    p_max: jnp.ndarray
    limit_type_mask_idxs: jdc.Static[tuple[int, ...]]

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the position barrier value.

        :param data: The MuJoCo simulation data.
        :return: The computed position barrier value.
        """
        obj_pos = self.get_pos(data)[self.mask_idxs,]
        return jnp.concatenate(
            [
                obj_pos - self.p_min,
                self.p_max - obj_pos,
            ]
        )[self.limit_type_mask_idxs,]


class PositionBarrier(ObjBarrier[JaxPositionBarrier]):
    """
    A position barrier class that wraps the JAX position barrier implementation.

    This class provides a high-level interface for position-specific barrier functions.

    :param p_min: The minimum allowed position.
    :param p_max: The maximum allowed position.
    :param limit_type: The type of limit to apply ('min', 'max', or 'both').
    """

    JaxComponentType: type = JaxPositionBarrier
    _p_min: jnp.ndarray
    _p_max: jnp.ndarray

    _limit_type: PositionLimitType

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        obj_name: str,
        obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY,
        p_min: ArrayOrFloat | None = None,
        p_max: ArrayOrFloat | None = None,
        limit_type: str = "both",
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        mask: Sequence[int] | None = None,
    ):
        """
        Initialize the PositionBarrier object.

        :param name: The name of the barrier.
        :param gain: The gain for the barrier function.
        :param obj_name: The name of the object (body, geometry, or site) to which this barrier applies.
        :param p_min: The minimum allowed position.
        :param p_max: The maximum allowed position.
        :param limit_type: The type of limit to apply ('min', 'max', or 'both').
        :param gain_fn: A function to compute the gain dynamically.
        :param safe_displacement_gain: The gain for computing safe displacements.
        :param mask: A sequence of integers to mask certain dimensions.
        """
        mask = mask if mask is not None else [1, 1, 1]
        super().__init__(name, gain, obj_name, obj_type, gain_fn, safe_displacement_gain, mask)
        if limit_type not in {"min", "max", "both"}:
            raise ValueError("[PositionBarrier] PositionBarrier.limit should be either 'min', 'max', or 'both'")

        # Setting up the dimension, using mask and limit type
        self._limit_type = PositionLimitType.from_str(limit_type)
        n_axes = len(self.mask_idxs)

        self._limit_type_mask_idxs: tuple[int, ...]
        match self.limit_type:
            case PositionLimitType.MIN:
                self._limit_type_mask_idxs = tuple(i for i in range(n_axes))
                self._dim = n_axes
            case PositionLimitType.MAX:
                self._limit_type_mask_idxs = tuple(i for i in range(n_axes, 2 * n_axes))
                self._dim = n_axes
            case PositionLimitType.BOTH:
                self._limit_type_mask_idxs = tuple(i for i in range(2 * n_axes))
                self._dim = 2 * n_axes

        self._p_min = jnp.zeros(n_axes)
        self._p_max = jnp.zeros(n_axes)
        if p_min is not None:
            self.update_p_min(p_min)

        if p_max is not None:
            self.update_p_max(p_max)

    @property
    def limit_type(self) -> PositionLimitType:
        """
        Get the type of limit applied to the position barrier.

        :return: The limit type.
        """
        return self._limit_type

    @property
    def limit_type_mask_idxs(self) -> tuple[int, ...]:
        return self._limit_type_mask_idxs

    @property
    def p_min(self) -> jnp.ndarray:
        """
        Get the minimum allowed position.

        :return: The minimum position.
        """
        return self._p_min

    @p_min.setter
    def p_min(self, value: ArrayOrFloat):
        """
        Set the minimum allowed position.

        :param value: The new minimum position.
        """

        self.update_p_min(value)

    def update_p_min(self, p_min: ArrayOrFloat):
        """
        Update the minimum allowed position.

        :param p_min: The new minimum position.
        :raises ValueError: If the dimension of p_min is incorrect.
        """

        p_min_jnp = jnp.array(p_min)
        if p_min_jnp.ndim == 0:
            p_min_jnp = jnp.ones(len(self.mask_idxs)) * p_min_jnp

        elif p_min_jnp.shape[-1] != len(self.mask_idxs):
            raise ValueError(
                f"[PositionBarrier] wrong dimension of p_min: expected {len(self.mask_idxs)}, got {p_min_jnp.shape[-1]}"
            )
        if not PositionLimitType.includes_min(self.limit_type):
            warnings.warn(
                "[PositionBarrier] type of the limit does not include minimum bound. Assignment is ignored",
                stacklevel=2,
            )
            return

        self._p_min = p_min_jnp

    @property
    def p_max(self) -> jnp.ndarray:
        """
        Get the maximum allowed position.

        :return: The maximum position.
        """
        return self._p_max

    @p_max.setter
    def p_max(self, value: ArrayOrFloat):
        self.update_p_max(value)

    def update_p_max(self, p_max: ArrayOrFloat):
        """
        Update the maximum allowed position.

        :param p_max: The new maximum position.
        :raises ValueError: If the dimension of p_max is incorrect.
        """
        p_max_jnp = jnp.array(p_max)
        if p_max_jnp.ndim == 0:
            p_max = jnp.ones(len(self.mask_idxs)) * p_max

        elif p_max_jnp.shape[-1] != len(self.mask_idxs):
            raise ValueError(
                f"[PositionBarrier] wrong dimension of p_max: expected {len(self.mask_idxs)}, got {p_max_jnp.shape[-1]}"
            )
        if not PositionLimitType.includes_max(self.limit_type):
            warnings.warn(
                "[PositionBarrier] type of the limit does not include maximum bound. Assignment is ignored",
                stacklevel=2,
            )
            return

        self._p_max = jnp.array(p_max)
