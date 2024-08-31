import abc
from typing import Callable, Self

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.configuration import update
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass(kw_only=True)
class JaxComponent(abc.ABC):
    dim: jdc.Static[int]
    model: mjx.Model
    gain: jnp.ndarray
    gain_function: jdc.Static[Callable[[float], float]]

    @abc.abstractmethod
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        pass

    def copy_and_set(self, **kwargs) -> Self:
        r"""..."""
        new_args = self.__dict__ | kwargs
        return self.__class__(**new_args)

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        return jax.jacrev(
            lambda q, model=self.model: self.__call__(
                update(model, q),
            ),
            argnums=0,
        )(data.qpos)


class Component[T: JaxComponent](abc.ABC):
    _dim: int
    __name: str
    __jax_component: T
    __model: mjx.Model | None
    __gain: jnp.ndarray
    __gain_fn: Callable[[float], float] | None
    _modified: bool

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
    ):
        self.__name = name
        self.__model = None
        self._modified = False

        self.gain = gain
        self.__gain_fn = gain_fn if gain_fn is not None else lambda x: x
        self._dim = -1

    @property
    def model(self) -> mjx.Model:
        if self.__model is None:
            raise ValueError("model is not provided yet")
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

    @property
    def vector_gain(self) -> jnp.ndarray:
        # scalar -> jnp.ones(self.dim) * scalar
        # vector -> vector
        if self._dim == -1:
            raise ValueError(
                "fail to calculate vector gain without dimension specified. "
                "You should provide robot model or pass component into the problem first."
            )

        match self.gain.ndim:
            case 0:
                return jnp.ones(self.dim) * self.gain
            case 1:
                if len(self.gain) != self.dim:
                    raise ValueError(f"invalid gain size: {self.gain.shape} != {self.dim}")
                return self.gain
            case _:
                raise ValueError("fail to construct vector gain from gain with ndim > 1")

    @gain.setter
    def gain(self, value: ArrayOrFloat):
        self.update_gain(value)

    def update_gain(self, gain: ArrayOrFloat):
        self._modified = True
        self.__gain = gain if isinstance(gain, jnp.ndarray) else jnp.array(gain)

    @property
    def gain_fn(self) -> Callable[[float], float]:
        return self.__gain_fn

    @property
    def name(self) -> str:
        return self.__name

    @property
    def dim(self) -> int:
        if self._dim == -1:
            raise ValueError(
                "component dimension is not defined yet. "
                "Provide robot model or pass component into the problem first."
            )
        return self._dim

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
