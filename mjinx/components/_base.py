from __future__ import annotations

import abc
from typing import Callable, Generic, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.configuration import update
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxComponent(abc.ABC):
    dim: jdc.Static[int]
    model: mjx.Model
    gain: jnp.ndarray
    gain_function: jdc.Static[Callable[[float], float]]
    mask_idxs: jdc.Static[tuple[int, ...]]

    @abc.abstractmethod
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        pass

    def copy_and_set(self, **kwargs) -> JaxComponent:
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


AtomicComponentType = TypeVar("AtomicComponentType", bound=JaxComponent)


class Component(Generic[AtomicComponentType], abc.ABC):
    _dim: int
    __name: str
    __jax_component: AtomicComponentType
    __model: mjx.Model | None
    __gain: jnp.ndarray
    __gain_fn: Callable[[float], float]
    __mask: jnp.ndarray | None
    __mask_idxs: tuple[int, ...]

    _modified: bool

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        mask: Sequence | None = None,
    ):
        self.__name = name
        self.__model = None
        self._modified = True

        self.update_gain(gain)
        self.__gain_fn = gain_fn if gain_fn is not None else lambda x: x
        self._dim = -1

        if mask is not None:
            self.__mask = jnp.array(mask)
            if self.__mask.ndim != 1:
                raise ValueError(f"mask is 1D vector, got {self.__mask.ndim}D array")
            self.__mask_idxs = tuple(i for i in range(len(self.__mask)) if self.__mask[i])
        else:
            self.__mask = None
            self.__mask_idxs = ()

    def _get_default_mask(self) -> tuple[jnp.ndarray, tuple[int, ...]]:
        return jnp.ones(self.dim), tuple(range(self.dim))

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

    @gain.setter
    def gain(self, value: ArrayOrFloat):
        self.update_gain(value)

    def update_gain(self, gain: ArrayOrFloat):
        gain = jnp.array(gain)
        if not isinstance(gain, float) and gain.ndim > 1:
            raise ValueError(f"gain ndim is too high: expected <= 1, got {gain.ndim}")
        self._modified = True
        self.__gain = gain

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

    @property
    def mask(self) -> jnp.ndarray:
        if self.__mask is None and self._dim == -1:
            raise ValueError("either mask should be provided explicitly, or dimension should be set")
        elif self.__mask is None:
            self.__mask, self.__mask_idxs = self._get_default_mask()

        return self.__mask

    @property
    def mask_idxs(self) -> tuple[int, ...]:
        if self.__mask is None and self._dim == -1:
            raise ValueError("either mask should be provided explicitly, or dimension should be set")
        elif self.__mask is None:
            self.__mask, self.__mask_idxs = self._get_default_mask()
        return self.__mask_idxs

    @abc.abstractmethod
    def _build_component(self) -> AtomicComponentType:
        pass

    @property
    def jax_component(self) -> AtomicComponentType:
        if self.__model is None:
            raise ValueError("model is not provided")
        if self._dim == -1:
            raise ValueError("dimension is not specified")

        if self._modified:
            self._modified = False
            self.__jax_component: AtomicComponentType = self._build_component()
        return self.__jax_component
