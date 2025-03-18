from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.configuration import jac_dq2v
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxComponent(abc.ABC):
    """Base class for all JAX-based components in the inverse kinematics framework.

    This class provides the fundamental structure for components that can be
    evaluated and differentiated within the JAX ecosystem. Components represent
    either tasks (objectives) or barriers (constraints) in the optimization problem.

    :param gain: The gain factor applied to the component.
    :param gain_fn: A function to compute the gain dynamically.
    :param mask: A sequence of integers to mask certain dimensions.
    """

    dim: jdc.Static[int]
    model: mjx.Model
    vector_gain: jnp.ndarray
    gain_fn: jdc.Static[Callable[[float], float]]
    mask_idxs: jdc.Static[tuple[int, ...]]

    def __post_init__(self):
        jdc._copy_and_mutate._mark_mutable(self, jdc._copy_and_mutate._Mutability.MUTABLE_NO_VALIDATION, visited=set())
        self._jac_fn = jax.jacrev(
            lambda q, model=self.model: self.__call__(
                mjx.kinematics(model, mjx.make_data(model).replace(qpos=q)),
            ),
            argnums=0,
        )
        jdc._copy_and_mutate._mark_mutable(self, jdc._copy_and_mutate._Mutability.FROZEN, visited=set())

    @abc.abstractmethod
    def __call__(self, data: mjx.Data) -> jnp.ndarray:  # pragma: no cover
        """
        Compute the component's value :math:`f(q, t)`.

        This method should be implemented by subclasses to provide specific
        component calculations.

        :param data: The MuJoCo simulation data.
        :return: The computed component value.
        """
        pass

    def copy_and_set(self, **kwargs) -> JaxComponent:
        """
        Create a copy of the component with updated attributes.

        :param kwargs: Keyword arguments specifying the attributes to update.
        :return: A new instance of the component with updated attributes.
        """
        new_args = self.__dict__ | kwargs
        return self.__class__(**new_args)

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        r"""
        Compute the Jacobian of the component with respect to the joint positions.
        The jacobian is a matrix :math:`J \in R^{dim \times nv}` that is defined as:

        .. math::

            J(q, t) = \frac{\partial f}{\partial q}

        The main property that is desirable for us is that given this matrix, we obtain
        component derivative as a linear matrix w.r.t. velocities:

        .. math::

            \dot{f}(q, t) = \frac{\partial f}{\partial q} v = J(q, t)


        The jacobian is calculated via automatic differentiation, if it is not overwritten.
        If :math:`nq \\neq nv`, a special mapping is computed to transform :math:`J_{dq}` into :math:`J_v`.
        For details, see :py:meth:`mjinx.configuration.jac_dq2v`.

        :param data: The MuJoCo simulation data.
        :return: The computed Jacobian matrix.
        """
        jac = self._jac_fn(data.qpos)
        if self.model.nq != self.model.nv:
            jac = jac @ jac_dq2v(self.model, data.qpos)
        return jac


AtomicComponentType = TypeVar("AtomicComponentType", bound=JaxComponent)


class Component(Generic[AtomicComponentType]):
    """High-level interface for components in the inverse kinematics framework.

    This class provides a Python-friendly interface for creating and manipulating
    components, which are then compiled into JAX-compatible representations for
    efficient computation.

    :param name: The name of the component.
    :param gain: The gain factor applied to the component.
    :param gain_fn: A function to compute the gain dynamically.
    :param mask: A sequence of integers to mask certain dimensions.
    """

    JaxComponentType: type

    _dim: int
    _name: str
    _jax_component: AtomicComponentType
    _model: mjx.Model | None
    _gain: jnp.ndarray
    _gain_fn: Callable[[float], float]
    _mask: jnp.ndarray | None
    _mask_idxs: tuple[int, ...] | jnp.ndarray

    __modified: bool

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        mask: Sequence[int] | None = None,
    ):
        """
        Initialize the Component object.

        :param name: The name of the component.
        :param gain: The gain for the component.
        :param gain_fn: A function to compute the gain dynamically.
        :param mask: A sequence of integers to mask certain dimensions.
        """
        self._name = name
        self._model = None

        self.update_gain(gain)
        self._gain_fn = gain_fn if gain_fn is not None else lambda x: x
        self._dim = -1

        if mask is not None:
            self._mask = jnp.array(mask)
            if self._mask.ndim != 1:
                raise ValueError(f"mask is 1D vector, got {self._mask.ndim}D array")
            self._mask_idxs = tuple(i for i in range(len(self._mask)) if self._mask[i])
        else:
            self._mask = None
            self._mask_idxs = ()

        self._jax_component = self.JaxComponentType(
            **{attr: None for attr in self.JaxComponentType.__dataclass_fields__.keys()}
        )  # type: ignore
        self._jax_component = jdc.replace(self._jax_component, gain_fn=self._gain_fn, mask_idxs=self._mask_idxs)

    def _get_default_mask(self) -> tuple[jnp.ndarray, tuple[int, ...]]:
        """
        Get the default mask for the component.

        :return: A tuple containing the default mask array and mask indices.
        """
        return jnp.ones(self.dim), tuple(range(self.dim))

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute of the component and mark it as modified.

        This method overrides the default attribute setting behavior to track
        when the component has been modified. This is useful for lazy evaluation and caching strategies.
        """

        super().__setattr__(name, value)
        if name != "_Component__modified":
            self.__modified = True

    @property
    def modified(self) -> bool:
        """
        Check if the component has been modified.

        :return: True if the component has been modified, False otherwise.
        """
        return self.__modified

    @property
    def model(self) -> mjx.Model:
        """
        Get the MuJoCo model associated with the component.

        :return: The MuJoCo model.
        """
        return self._model

    @model.setter
    def model(self, value: mjx.Model):
        """
        Set the MuJoCo model for the component.

        :param value: The MuJoCo model to set.
        """
        self.update_model(value)
        self._jax_component = jdc.replace(
            self._jax_component,
            model=self._model,
            vector_gain=self.vector_gain,
            mask_idxs=self.mask_idxs,
        )

    def update_model(self, model: mjx.Model):
        """
        Update the MuJoCo model for the component.

        :param model: The new MuJoCo model.
        """
        self._model = model

    @property
    def gain(self) -> jnp.ndarray:
        """
        Get the gain for the component.

        :return: The gain array.
        """
        return self._gain

    @gain.setter
    def gain(self, value: ArrayOrFloat):
        """
        Set the gain for the component.

        :param value: The new gain value.
        """
        self.update_gain(value)

    def update_gain(self, gain: ArrayOrFloat):
        """
        Update the gain for the component.

        :param gain: The new gain value.
        :raises ValueError: If the gain has an invalid dimension.
        """
        gain = jnp.array(gain)
        if not isinstance(gain, float) and gain.ndim > 1:
            raise ValueError(f"gain ndim is too high: expected <= 1, got {gain.ndim}")
        self._gain = gain

    @property
    def vector_gain(self) -> jnp.ndarray:
        """
        Get the vector gain for the component.

        The vector gain is generated as follows:
            - gain is scalar -> vector_gain = jnp.ones(self.dim) * gain
            - gain is vector -> vector_gain = gain

        :return: The vector gain array.
        :raises ValueError: If the dimension is not set or the gain size is invalid.
        """
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
            case _:  # pragma: no cover
                raise ValueError("fail to construct vector gain from gain with ndim > 1")

    @property
    def gain_fn(self) -> Callable[[float], float]:
        """
        Get the gain function for the component.

        :return: The gain function.
        """
        return self._gain_fn

    @property
    def name(self) -> str:
        """
        Get the name of the component.

        :return: The component name.
        """
        return self._name

    @property
    def dim(self) -> int:
        """
        Get the dimension of the component.

        :return: The component dimension.
        :raises ValueError: If the dimension is not set.
        """
        if self._dim == -1:
            raise ValueError(
                "component dimension is not defined yet. Provide robot model or pass component into the problem first."
            )
        return self._dim

    @property
    def mask(self) -> jnp.ndarray:
        """
        Get the mask for the component.

        :return: The mask array.
        :raises ValueError: If the mask is not set and the dimension is not defined.
        """
        if self._mask is None and self._dim == -1:
            raise ValueError("either mask should be provided explicitly, or dimension should be set")
        elif self._mask is None:
            self._mask, _ = self._get_default_mask()

        return self._mask

    @property
    def mask_idxs(self) -> tuple[int, ...]:
        """
        Get the mask indices for the component.

        :return: A tuple of mask indices.
        :raises ValueError: If the mask is not set and the dimension is not defined.
        """
        if self._mask_idxs is None and self._dim == -1:
            raise ValueError("either mask should be provided explicitly, or dimension should be set")
        elif self._mask_idxs is None:
            _, self._mask_idxs = self._get_default_mask()
        return tuple(self._mask_idxs)

    def _build_component(self) -> AtomicComponentType:
        """
        Build the atomic component implementation.

        :return: An instance of the atomic component.
        """
        component_attributes = self.JaxComponentType.__dataclass_fields__.keys()  # type: ignore
        return self.JaxComponentType(**{attr: self.__getattribute__(attr) for attr in component_attributes})

    @property
    def jax_component(self) -> AtomicComponentType:
        """
        Get the JAX implementation of the component.

        :return: The JAX component instance.
        :raises ValueError: If the model or dimension is not set.
        """
        if self._model is None:
            raise ValueError("model is not provided")
        if self._dim == -1:
            raise ValueError("dimension is not specified")

        # if self.__modified:
        #     self._jax_component: AtomicComponentType = self._build_component()
        #     self.__modified = False
        return self._jax_component
