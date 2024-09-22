from __future__ import annotations

from enum import Enum
from typing import Callable, TypeAlias, TypedDict

import jax.numpy as jnp
import numpy as np

ndarray: TypeAlias = np.ndarray | jnp.ndarray
ArrayOrFloat: TypeAlias = ndarray | float
ClassKFunctions: TypeAlias = Callable[[ndarray], ndarray]
CallableFromState: TypeAlias = Callable[[ndarray, ndarray], ndarray]
FrictionCallable: TypeAlias = Callable[[ndarray, ndarray], ndarray]

# Type aliases.
CollisionBody = int | str
CollisionPair = tuple[int, int]


class InterfaceVariableParameters(TypedDict, total=False):
    min: np.ndarray
    max: np.ndarray
    is_hard: bool
    activation: bool
    weight: float
    reference: np.ndarray


class PositionLimitType(Enum):
    MIN = 0
    MAX = 1
    BOTH = 2

    @staticmethod
    def from_str(type: str) -> PositionLimitType:
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
        return type == PositionLimitType.MIN or type == PositionLimitType.BOTH

    @staticmethod
    def includes_max(type: PositionLimitType) -> bool:
        return type == PositionLimitType.MAX or type == PositionLimitType.BOTH
