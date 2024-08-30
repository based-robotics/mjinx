from typing import cast

import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components._base import Component, JaxComponent
from mjinx.components.barriers._base import Barrier
from mjinx.components.tasks._base import Task


@jdc.pytree_dataclass
class JaxProblemData:
    model: mjx.Model
    components: dict[str, JaxComponent]


class Problem:
    __model: mjx.Model
    __components: dict[str, Component]

    def __init__(self, model: mjx.Model):
        self.__model = model
        self.__components = {}

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
        return JaxProblemData(self.__model, components)

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
