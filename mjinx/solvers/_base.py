import abc

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx import configuration
from mjinx.problem import JaxProblemData


@jdc.pytree_dataclass
class SolverState:
    pass


class Solver[S: SolverState](abc.ABC):
    model: mjx.Model

    def __init__(self, model: mjx.Model | None = None):
        self.model = model

    @abc.abstractmethod
    def solve_from_data(self, data: mjx.Data, solver_state: S, components: JaxProblemData) -> tuple[jnp.ndarray, S]:
        if self.model is None:
            raise ValueError("model is not provided, solution is unavailable")

    def solve(self, q: jnp.ndarray, solver_data: S, problem_data: JaxProblemData) -> tuple[jnp.ndarray, S]:
        return self.solve_from_data(configuration.update(self.model, q), solver_data, problem_data)

    @abc.abstractmethod
    def init(self, data: jnp.ndarray) -> S:
        pass
