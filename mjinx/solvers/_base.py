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
    """Base class for solver-specific data.

    This class serves as a placeholder for any data that a specific solver might need to maintain
    between iterations or function calls.
    """

    pass


@jdc.pytree_dataclass
class SolverSolution:
    """Base class for solver solutions.

    This class serves as a placeholder for any output solution that a specific solver would return
    as a result.

    :param v_opt: Optimal velocity solution.
    """

    v_opt: jnp.ndarray


SolverDataType = TypeVar("SolverDataType", bound=SolverData)
SolverSolutionType = TypeVar("SolverSolutionType", bound=SolverSolution)


class Solver(Generic[SolverDataType, SolverSolutionType], abc.ABC):
    """Abstract base class for solvers.

    This class defines the interface for solvers used in inverse kinematics problems.

    :param model: The MuJoCo model used by the solver.
    """

    model: mjx.Model

    def __init__(self, model: mjx.Model):
        """Initialize the solver with a MuJoCo model.

        :param model: The MuJoCo model to be used by the solver.
        """
        self.model = model

    @abc.abstractmethod
    def solve_from_data(
        self, solver_data: SolverDataType, problem_data: JaxProblemData, model_data: mjx.Data
    ) -> tuple[SolverSolutionType, SolverDataType]:
        """Solve the inverse kinematics problem using pre-computed updateddata.

        :param solver_data: Solver-specific data.
        :param problem_data: Problem-specific data.
        :param model_data: MuJoCo model data.
        :return: A tuple containing the solver solution and updated solver data.
        """
        pass

    def solve(
        self, q: jnp.ndarray, solver_data: SolverDataType, problem_data: JaxProblemData
    ) -> tuple[SolverSolutionType, SolverDataType]:
        """Solve the inverse kinematics problem for a given configuration.

        This method creates mjx.Data instance and updates it under the hood. To avoid doing an extra
        update, consider solve_from_data method.

        :param q: The current joint configuration.
        :param solver_data: Solver-specific data.
        :param problem_data: Problem-specific data.
        :return: A tuple containing the solver solution and updated solver data.
        :raises ValueError: If the input configuration has incorrect dimensions.
        """
        if q.shape != (self.model.nq,):
            raise ValueError(f"wrong dimension of the state: expected ({self.model.nq}, ), got {q.shape}")
        model_data = configuration.update(self.model, q)
        return self.solve_from_data(solver_data, problem_data, model_data)

    @abc.abstractmethod
    def init(self, q: mjt.ndarray) -> SolverDataType:
        """Initialize solver-specific data.

        :param q: The initial joint configuration.
        :return: Initialized solver-specific data.
        """
        pass
