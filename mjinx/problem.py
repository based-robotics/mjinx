# problem.py

from contextlib import AbstractContextManager
from typing import cast

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components._base import Component, JaxComponent
from mjinx.components.barriers._base import Barrier
from mjinx.components.tasks._base import Task
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxProblemData:
    """
    A dataclass representing the compiled problem data for JAX optimization.

    :param model: The MuJoCo model.
    :param v_min: Minimum velocity limits.
    :param v_max: Maximum velocity limits.
    :param components: Dictionary of JAX components.
    """

    model: mjx.Model
    v_min: jnp.ndarray
    v_max: jnp.ndarray
    components: dict[str, JaxComponent]


class Problem:
    """
    A class representing an optimization problem for robotics applications.

    This class manages the components, velocity limits, and model for the optimization problem.

    :param model: The MuJoCo model.
    :param v_min: Minimum velocity limit (default: 1e-3).
    :param v_max: Maximum velocity limit (default: 1e3).
    """

    def __init__(self, model: mjx.Model, v_min: ArrayOrFloat = -1e3, v_max: ArrayOrFloat = 1e3):
        self.__model = model
        self.__components: dict[str, Component] = {}
        self.update_v_min(v_min)
        self.update_v_max(v_max)

    @property
    def v_min(self) -> jnp.ndarray:
        """
        Get the minimum velocity limits.

        :return: The minimum velocity limits.
        """
        return self.__v_min

    @v_min.setter
    def v_min(self, v_min: ArrayOrFloat):
        """
        Set the minimum velocity limits.

        :param v_min: The new minimum velocity limits.
        """
        self.update_v_min(v_min)

    def update_v_min(self, v_min: ArrayOrFloat):
        """
        Update the minimum velocity limits.

        :param v_min: The new minimum velocity limits.
        :raises ValueError: If the shape of v_min is invalid.
        """
        v_min_jnp: jnp.ndarray = jnp.array(v_min)
        match v_min_jnp.ndim:
            case 0:
                self.__v_min = jnp.ones(self.__model.nv) * v_min_jnp
            case 1:
                if v_min_jnp.shape != (self.__model.nv,):
                    raise ValueError(f"invalid v_min shape: expected ({self.__model.nv},) got {v_min_jnp.shape}")
                self.__v_min = v_min_jnp
            case _:
                raise ValueError("v_min with ndim>1 is not allowed")

    @property
    def v_max(self) -> jnp.ndarray:
        """
        Get the maximum velocity limits.

        :return: The maximum velocity limits.
        """
        return self.__v_max

    @v_max.setter
    def v_max(self, v_max: ArrayOrFloat):
        """
        Set the maximum velocity limits.

        :param v_max: The new maximum velocity limits.
        """
        self.update_v_max(v_max)

    def update_v_max(self, v_max: ArrayOrFloat):
        """
        Update the maximum velocity limits.

        :param v_max: The new maximum velocity limits.
        :raises ValueError: If the shape of v_max is invalid.
        """
        v_max_jnp: jnp.ndarray = jnp.array(v_max)
        match v_max_jnp.ndim:
            case 0:
                self.__v_max = jnp.ones(self.__model.nv) * v_max_jnp
            case 1:
                if v_max_jnp.shape != (self.__model.nv,):
                    raise ValueError(f"invalid v_max shape: expected ({self.__model.nv},) got {v_max_jnp.shape}")
                self.__v_max = v_max_jnp
            case _:
                raise ValueError("v_max with ndim>1 is not allowed")

    def add_component(self, component: Component):
        """
        Add a component to the problem.

        :param component: The component to add.
        :raises ValueError: If a component with the same name already exists.
        """
        if component.name in self.__components:
            raise ValueError("the component with this name already exists")
        component.model = self.__model
        self.__components[component.name] = component

    def remove_component(self, name: str):
        """
        Remove a component from the problem.

        :param name: The name of the component to remove.
        """
        if name in self.__components:
            del self.__components[name]

    def compile(self) -> JaxProblemData:
        """
        Compile the problem into a JaxProblemData object.

        :return: A JaxProblemData object containing the compiled problem data.
        """
        components = {
            name: cast(JaxComponent, component.jax_component) for name, component in self.__components.items()
        }
        return JaxProblemData(self.__model, self.v_min, self.v_max, components)  # type: ignore[call-arg]

    def component(self, name: str) -> Component:
        """
        Get a component by name.

        :param name: The name of the component.
        :return: The requested component.
        :raises ValueError: If the component is not found.
        """
        if name not in self.__components:
            raise ValueError("component is not present in the dictionary")
        return self.__components[name]

    def task(self, name: str) -> Task:
        """
        Get a task component by name.

        :param name: The name of the task component.
        :return: The requested task component.
        :raises ValueError: If the component is not found or is not a task.
        """
        if name not in self.__components:
            raise ValueError("component is not present in the dictionary")
        if not isinstance(self.__components[name], Task):
            raise ValueError("specified component is not a task")
        return cast(Task, self.__components[name])

    def barrier(self, name: str) -> Barrier:
        """
        Get a barrier component by name.

        :param name: The name of the barrier component.
        :return: The requested barrier component.
        :raises ValueError: If the component is not found or is not a barrier.
        """
        if name not in self.__components:
            raise ValueError("component is not present in the dictionary")
        if not isinstance(self.__components[name], Barrier):
            raise ValueError("specified component is not a barrier")
        return cast(Barrier, self.__components[name])

    def set_vmap_dimension(self) -> AbstractContextManager[JaxProblemData]:
        """
        Set the vmap dimension for the problem.

        :return: A context manager for the JaxProblemData with vmap dimension set.
        """
        return jdc.copy_and_mutate(jax.tree_util.tree_map(lambda x: None, self.compile()), validate=False)
