from typing import ContextManager, cast

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
    # TODO: should v_min and v_max be here?,.
    model: mjx.Model
    v_min: jnp.ndarray
    v_max: jnp.ndarray
    components: dict[str, JaxComponent]


class Problem:
    __model: mjx.Model
    __components: dict[str, Component]
    __v_min: jnp.ndarray
    __v_max: jnp.ndarray

    def __init__(self, model: mjx.Model, v_min: ArrayOrFloat, v_max: ArrayOrFloat):
        self.__model = model
        self.__components = {}
        self.v_min = v_min
        self.v_max = v_max

    @property
    def v_min(self) -> jnp.ndarray:
        return self.__v_min

    @v_min.setter
    def v_min(self, v_min: ArrayOrFloat):
        if not isinstance(v_min, jnp.ndarray):
            v_min = jnp.array(v_min)
        match v_min.ndim:
            case 0:
                self.__v_min = jnp.ones(self.__model.nv) * v_min
            case 1:
                if v_min.shape != (self.__model.nv,):
                    raise ValueError(f"invalid v_min shape: expected ({self.__model.nv},) got {v_min.shape}")
                self.__v_min = v_min
            case _:
                raise ValueError("v_min with ndim>1 is not allowed")

    @property
    def v_max(self) -> jnp.ndarray:
        return self.__v_max

    @v_max.setter
    def v_max(self, v_max: ArrayOrFloat):
        if not isinstance(v_max, jnp.ndarray):
            v_max = jnp.array(v_max)
        match v_max.ndim:
            case 0:
                self.__v_max = jnp.ones(self.__model.nv) * v_max
            case 1:
                if v_max.shape != (self.__model.nv,):
                    raise ValueError(f"invalid v_max shape: expected ({self.__model.nv},) got {v_max.shape}")
                self.__v_max = v_max
            case _:
                raise ValueError("v_max with ndim>1 is not allowed")

    def add_component(self, component: Component):
        if component.name in self.__components:
            raise ValueError("the component with this name already exists")
        component.model = self.__model
        self.__components[component.name] = component

    def remove_component(self, name: str):
        if name in self.__components:
            del self.__components[name]

    def compile(self) -> JaxProblemData:
        # TODO: is this cast is considered a normal practice?..
        components = {
            name: cast(JaxComponent, component.jax_component) for name, component in self.__components.items()
        }
        return JaxProblemData(self.__model, self.v_min, self.v_max, components)

    def component(self, name: str) -> Component:
        if name not in self.__components:
            raise ValueError("component is not present in the dictionary")
        return self.__components[name]

    # TODO: should specific getters for task and barrier even be here?..
    def task(self, name: str) -> Task:
        if name not in self.__components:
            raise ValueError("component is not present in the dictionary")
        if not isinstance(self.__components[name], Task):
            raise ValueError("specified component is not a task")
        return cast(Barrier, self.__components[name])

    def barrier(self, name: str) -> Barrier:
        if name not in self.__components:
            raise ValueError("component is not present in the dictionary")
        if not isinstance(self.__components[name], Barrier):
            raise ValueError("specified component is not a barrier")
        return cast(Barrier, self.__components[name])

    def set_vmap_dimension(self) -> ContextManager[JaxProblemData]:
        return jdc.copy_and_mutate(jax.tree_util.tree_map(lambda x: None, self.compile()), validate=False)
