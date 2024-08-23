from typing import Callable, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

ndarray = np.ndarray | jnp.ndarray
ClassKFunctions = Callable[[ndarray], ndarray]
Gain = ndarray | float
CallableFromState = Callable[[ndarray, ndarray], ndarray]
FrictionCallable = Callable[[ndarray, ndarray], ndarray]


class InterfaceVariableParameters(TypedDict, total=False):
    min: np.ndarray
    max: np.ndarray
    is_hard: bool
    activation: bool
    weight: float
    reference: np.ndarray
