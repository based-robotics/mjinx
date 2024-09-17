from typing import Callable, TypeAlias, TypedDict

import jax.numpy as jnp
import numpy as np

ndarray: TypeAlias = np.ndarray | jnp.ndarray
ArrayOrFloat: TypeAlias = ndarray | float
ClassKFunctions: TypeAlias = Callable[[ndarray], ndarray]
CallableFromState: TypeAlias = Callable[[ndarray, ndarray], ndarray]
FrictionCallable: TypeAlias = Callable[[ndarray, ndarray], ndarray]


class InterfaceVariableParameters(TypedDict, total=False):
    min: np.ndarray
    max: np.ndarray
    is_hard: bool
    activation: bool
    weight: float
    reference: np.ndarray
