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

    This class serves as a placeholder for any data that a specific solver needs to maintain
    between iterations or function calls. It enables solver implementations to preserve
    state information and warm-start subsequent optimization steps.
    """

    pass


@jdc.pytree_dataclass
class SolverSolution:
    """Base class for solver solutions.

    This class provides the structure for returning optimization results. It contains
    the optimal velocity solution and can be extended by specific solvers to include
    additional solution information.

    :param v_opt: Optimal velocity solution.
    """

    v_opt: jnp.ndarray


SolverDataType = TypeVar("SolverDataType", bound=SolverData)
SolverSolutionType = TypeVar("SolverSolutionType", bound=SolverSolution)


class Solver(Generic[SolverDataType, SolverSolutionType], abc.ABC):
    r"""Abstract base class for solvers.

    This class defines the interface for solvers used in inverse kinematics problems.
    Solvers transform task and barrier constraints into optimization problems, which
    can be solved to find joint configurations or velocities that satisfy the constraints.

    In general, the optimization problem can be formulated as:

    .. math::

        \min_{q} \sum_{i} \|e_i(q)\|^2_{W_i} \quad \text{subject to} \quad h_j(q) \geq 0

    where:
        - :math:`e_i(q)` are the task errors
        - :math:`W_i` are the task weight matrices
        - :math:`h_j(q)` are the barrier constraints

    Different solver implementations use different approaches to solve this problem,
    such as local linearization (QP) or global nonlinear optimization.

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
        self, model_data: mjx.Data, solver_data: SolverDataType, problem_data: JaxProblemData
    ) -> tuple[SolverSolutionType, SolverDataType]:
        """Solve the inverse kinematics problem using pre-computed model data.

        :param model_data: MuJoCo model data.
        :param solver_data: Solver-specific data.
        :param problem_data: Problem-specific data.
        :return: A tuple containing the solver solution and updated solver data.
        """
        pass

    def solve(
        self, q: jnp.ndarray, model_data: mjx.Data, solver_data: SolverDataType, problem_data: JaxProblemData
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
        model_data = configuration.update(self.model, model_data.replace(qpos=q))
        return self.solve_from_data(model_data, solver_data, problem_data)

    @abc.abstractmethod
    def init(self, q: mjt.ndarray) -> SolverDataType:
        """Initialize solver-specific data.

        :param q: The initial joint configuration.
        :return: Initialized solver-specific data.
        """
        pass
