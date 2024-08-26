import abc
from typing import Any

import jax.numpy as jnp
import mujoco.mjx as mjx

from mjinx import configuration
from mjinx.components._base import Component

# TODO: replace any type with solution data type
SolverData = Any


class Solver:
    __model: mjx.Model
    __components: dict[str, Component]

    def __init__(self, model: mjx.Model):
        self.__model = model

    def add_component(self, component: Component):
        if component.__name in self.__components:
            raise ValueError("the component with this name already exists")
        component.model = self.__model
        self.__components[component.name]

    @abc.abstractmethod
    def solve_from_data(self, data: mjx.Data, damping: float) -> tuple[jnp.ndarray, SolverData]:
        pass

    def solve(self, q: jnp.ndarray, damping: float) -> tuple[jnp.ndarray, SolverData]:
        self.solve_from_data(configuration.update(self.model, q))

    def solve_on_horizon(
        self,
        q0: jnp.ndarray,
        dt: float,
        horizon: int,
        damping: float = 1e-12,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""..."""
        qs = [q0]
        vs = []
        data = configuration.update(self.model, q0)
        for _ in range(horizon):
            v, _ = self.solve_from_data(data, damping)
            data = configuration.integrate_inplace(self.model, data, v, dt)

            qs.append(data.qpos)
            vs.append(v)

        return jnp.stack(qs), jnp.stack(vs)
