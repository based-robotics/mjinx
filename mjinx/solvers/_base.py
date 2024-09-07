import abc
from typing import Generic, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

import mjinx.typing as mjt
from mjinx import configuration
from mjinx.problem import JaxProblemData


@jdc.pytree_dataclass
class SolverData:
    pass


@jdc.pytree_dataclass
class SolverSolution:
    v_opt: jnp.ndarray


SolverDataType = TypeVar("SolverDataType", bound=SolverData)
SolverSolutionType = TypeVar("SolverSolutionType", bound=SolverSolution)


class Solver(Generic[SolverDataType, SolverSolutionType], abc.ABC):
    model: mjx.Model

    def __init__(self, model: mjx.Model | None = None):
        self.model = model

    @abc.abstractmethod
    def solve_from_data(
        self, solver_data: SolverDataType, problem_data: JaxProblemData, model_data: mjx.Data
    ) -> tuple[SolverSolutionType, SolverDataType]:
        pass

    def solve(
        self, q: jnp.ndarray, solver_data: SolverDataType, problem_data: JaxProblemData
    ) -> tuple[SolverSolutionType, SolverDataType]:
        if self.model is None:
            raise ValueError("model is not provided, solution is unavailable")
        model_data = configuration.update(self.model, q)
        return self.solve_from_data(solver_data, problem_data, model_data)

    @abc.abstractmethod
    def init(self, q: mjt.ndarray) -> SolverDataType:
        pass
