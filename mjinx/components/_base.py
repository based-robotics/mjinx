import abc
from dataclasses import field
from typing import Callable, ClassVar, Self

import jax
import jax.numpy as jnp
import numpy as np
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.configuration import update
from mjinx.typing import Gain


@jdc.pytree_dataclass(kw_only=True)
class JaxComponent(abc.ABC):
    dim: ClassVar[int]

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


class Component(abc.ABC):
    __jax_component: JaxComponent
    __model: mjx.Model
    __gain: jnp.ndarray
    __gain_fn: Callable[[float, float]] | None
    __modified: bool

    def __init__(self, model: mjx.Model, gain: Gain, gain_fn: Callable[[float, float]] | None = None):
        self.__modified = False
        self.model = model
        self.gain = gain
        self.__gain_fn = gain_fn if gain_fn is not None else lambda x: x

    @property
    def model(self) -> mjx.Model:
        return self.__model

    @model.setter
    def model(self, value: mjx.Model):
        self.update_model(value)

    def update_model(self, model: mjx.Model):
        self.__modified = True
        self.__model = model

    @property
    def gain(self) -> jnp.ndarray:
        return self.__gain

    @gain.setter
    def gain(self, value: Gain):
        self.update_gain(value)

    def update_gain(self, gain: Gain):
        self.__modified = True
        self.__gain = gain if isinstance(gain, jnp.ndarray) else jnp.ndarray(gain)

    @property
    def jax_component(self) -> JaxComponent:
        if self.__modified:
            self.__modified = False
            self.__jax_component = JaxComponent(
                model=self.__model,
                gain=self.__gain,
                gain_fn=self.__gain_fn,
            )
        return self.__jax_component
