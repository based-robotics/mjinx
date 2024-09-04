import abc

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx import configuration
from mjinx.problem import JaxProblemData


@jdc.pytree_dataclass
class SolverData:
    pass


@jdc.pytree_dataclass
class SolverSolution:
    v_opt: jnp.ndarray


class Solver[D: SolverData, S: SolverSolution](abc.ABC):
    model: mjx.Model

    def __init__(self, model: mjx.Model | None = None):
        self.model = model

    @abc.abstractmethod
    def solve_from_data(self, solver_data: D, problem_data: JaxProblemData, model_data: mjx.Data) -> tuple[S, D]:
        if self.model is None:
            raise ValueError("model is not provided, solution is unavailable")

    def solve(self, q: jnp.ndarray, solver_data: D, problem_data: JaxProblemData) -> tuple[S, D]:
        model_data = configuration.update(self.model, q)
        return self.solve_from_data(solver_data, problem_data, model_data)

    @abc.abstractmethod
    def init(self) -> D:
        pass
