import abc
from dataclasses import InitVar, field
from typing import ClassVar, Self
from ..barriers import Barrier
from ..configuration import get_configuration_limit, update
from ..tasks import Task


import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from ..configuration import update


class Solver:
    model: mjx.Model
    data: mjx.Model
    __tasks: dict[str, Task]
    __barriers: dict[str, Barrier]

    def __init__(self):
        pass

    @abc.abstractmethod
    def solve(self, q: jnp.ndarray):
        pass

    def task(self, name: str) -> Task:
        return self.__tasks[name]

    def barrier(self, name: str) -> Barrier:
        return self.__barriers[name]

    def update_task(self, name: str, **kwargs):
        self.__tasks[name] = self.__tasks[name].copy_and_set(**kwargs)

    def update_barrier(self, name: str, **kwargs):
        self.__barriers[name] = self.__barriers[name].copy_and_set(**kwargs)
