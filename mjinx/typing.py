"""
This module contains type definitions and aliases used throughout the mjinx library.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import NamedTuple, TypeAlias

import jax.numpy as jnp
import numpy as np
from mujoco.mjx._src.dataclasses import PyTreeNode

ndarray: TypeAlias = np.ndarray | jnp.ndarray
"""Type alias for numpy or JAX numpy arrays."""

ArrayOrFloat: TypeAlias = ndarray | float
"""Type alias for an array or a float value."""

ClassKFunctions: TypeAlias = Callable[[ndarray], ndarray]
"""Type alias for Class K functions, which are scalar functions that take and return ndarrays."""

CollisionBody: TypeAlias = int | str
"""Type alias for collision body representation, either as an integer ID or a string name."""

CollisionPair: TypeAlias = tuple[int, int]
"""Type alias for a pair of collision body IDs."""


class SimplifiedContact(PyTreeNode):
    geom: jnp.ndarray
    dist: jnp.ndarray
    pos: jnp.ndarray
    frame: jnp.ndarray


class PositionLimitType(Enum):
    """Type which describes possible position limits.

    The position limit could be only minimal, only maximal, or minimal and maximal.
    """

    MIN = 0
    MAX = 1
    BOTH = 2

    @staticmethod
    def from_str(type: str) -> PositionLimitType:
        """Generates position limit type from string.

        :param type: position limit type.
        :raises ValueError: limit name is not 'min', 'max', or 'both'.
        :return: corresponding enum type.
        """
        match type.lower():
            case "min":
                return PositionLimitType.MIN
            case "max":
                return PositionLimitType.MAX
            case "both":
                return PositionLimitType.BOTH
            case _:
                raise ValueError(
                    f"[PositionLimitType] invalid position limit type: {type}. " f"Expected {{'min', 'max', 'both'}}"
                )

    @staticmethod
    def includes_min(type: PositionLimitType) -> bool:
        """Either given limit includes minimum limit or not.

        Returns true, if limit is either MIN or BOTH, and false otherwise.

        :param type: limit to be processes.
        :return: True, if limit includes minimum limit, False otherwise.
        """
        return type == PositionLimitType.MIN or type == PositionLimitType.BOTH

    @staticmethod
    def includes_max(type: PositionLimitType) -> bool:
        """Either given limit includes maximum limit or not.

        Returns true, if limit is either MIN or BOTH, and false otherwise.

        :param type: limit to be processes.
        :return: True, if limit includes maximum limit, False otherwise.
        """

        return type == PositionLimitType.MAX or type == PositionLimitType.BOTH
