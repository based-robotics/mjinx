import abc
from typing import Callable, Self

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.configuration import update
from mjinx.typing import Gain


@jdc.pytree_dataclass(kw_only=True)
class JaxComponent(abc.ABC):
    dim: int
    model: mjx.Model
    gain: jnp.ndarray
    gain_function: jdc.Static[Callable[[float], float]]

    @abc.abstractmethod
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def compute_qp_objective(self, data: mjx.Data) -> tuple[jnp.ndarray, jnp.ndarray]:
        r""""""
        pass

    @abc.abstractmethod
    def compute_qp_inequality(
        self,
        data: mjx.Data,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r""""""

    def copy_and_set(self, **kwargs) -> Self:
        r"""..."""
        new_args = self.__dict__ | kwargs
        return self.__class__(**new_args)

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        return jax.jacrev(
            lambda q, model=self.model: self.__call__(
                update(model, qpos=q),
            ),
            argnums=0,
        )(data.qpos)


class Component[T: JaxComponent](abc.ABC):
    __name: str
    __jax_component: T
    __model: mjx.Model
    __gain: jnp.ndarray
    __gain_fn: Callable[[float, float]] | None
    _modified: bool

    def __init__(self, name: str, gain: Gain, gain_fn: Callable[[float, float]] | None = None):
        self.__name = name
        self.__model = None
        self._modified = False

        self.gain = gain
        self.__gain_fn = gain_fn if gain_fn is not None else lambda x: x

    @property
    def model(self) -> mjx.Model:
        return self.__model

    @model.setter
    def model(self, value: mjx.Model):
        self.update_model(value)

    def update_model(self, model: mjx.Model):
        self._modified = True
        self.__model = model

    @property
    def gain(self) -> jnp.ndarray:
        return self.__gain

    @gain.setter
    def gain(self, value: Gain):
        self.update_gain(value)

    def update_gain(self, gain: Gain):
        self._modified = True
        self.__gain = gain if isinstance(gain, jnp.ndarray) else jnp.ndarray(gain)

    @property
    def gain_fn(self) -> callable[[float], float]:
        return self.__gain_fn

    @property
    def name(self) -> str:
        return self.__name

    @abc.abstractmethod
    def _build_component(self) -> T:
        if self.__model is None:
            raise ValueError("model is not provided")

    @property
    def jax_component(self) -> T:
        if self._modified:
            self._modified = False
            self.__jax_component: T = self._build_component()
        return self.__jax_component
