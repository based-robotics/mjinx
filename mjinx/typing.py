"""Typings which are utilized in the mjinx"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TypeAlias

import jax.numpy as jnp
import numpy as np

ndarray: TypeAlias = np.ndarray | jnp.ndarray
ArrayOrFloat: TypeAlias = ndarray | float
# Class K function is a scalar function
ClassKFunctions: TypeAlias = Callable[[ndarray], ndarray]

CollisionBody = int | str  # body id, or body name
CollisionPair = tuple[int, int]  # pair body ids


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
