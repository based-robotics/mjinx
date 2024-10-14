import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import optax
from chex import ArrayTree  # noqa: F401

import mjinx.typing as mjt
from mjinx import configuration
from mjinx.components.barriers._base import JaxBarrier
from mjinx.components.tasks._base import JaxTask
from mjinx.problem import JaxProblemData
from mjinx.solvers._base import Solver, SolverData, SolverSolution


@jdc.pytree_dataclass
class GlobalIKData(SolverData):
    """Data class for the Global Inverse Kinematics solver.

    :param optax_state: The state of the Optax optimizer.
    """

    optax_state: optax.OptState


@jdc.pytree_dataclass
class GlobalIKSolution(SolverSolution):
    """Solution class for the Global Inverse Kinematics solver.

    :param q_opt: The optimal joint configuration.
    :param v_opt: The optimal joint velocities.
    """

    q_opt: jnp.ndarray


class GlobalIKSolver(Solver[GlobalIKData, GlobalIKSolution]):
    """Global Inverse Kinematics solver using gradient-based optimization.

    This solver uses Optax for gradient-based optimization to solve the inverse kinematics problem globally.

    :param model: The MuJoCo model.
    :param optimizer: The Optax optimizer to use.
    :param dt: The time step for velocity computation.
    """

    def __init__(self, model: mjx.Model, optimizer: optax.GradientTransformation, dt: float = 1e-2):
        """Initialize the Global IK solver.

        :param model: The MuJoCo model.
        :param optimizer: The Optax optimizer to use.
        :param dt: The time step for velocity computation.
        """
        super().__init__(model)
        self._optimizer = optimizer
        self.grad_fn = jax.grad(
            self.loss_fn,
            argnums=0,
        )
        self.__dt = dt

    def __log_barrier(self, x: jnp.ndarray, gain: jnp.ndarray):
        """Compute the logarithmic barrier function.

        :param x: The input values.
        :param gain: The gain values for the barrier.
        :return: The computed logarithmic barrier value.
        """
        return jnp.sum(gain * jax.lax.map(jnp.log, x))

    def loss_fn(self, q: jnp.ndarray, problem_data: JaxProblemData) -> float:
        """Compute the loss function for the given joint configuration.

        :param q: The joint configuration.
        :param problem_data: The problem-specific data.
        :return: The computed loss value.
        """
        model_data = configuration.update(problem_data.model, q)
        loss = 0

        for component in problem_data.components.values():
            if isinstance(component, JaxTask):
                err = component(model_data)
                loss = loss + component.vector_gain * err.T @ err  # type: ignore
            if isinstance(component, JaxBarrier):
                loss = loss - self.__log_barrier(jnp.clip(component(model_data), 1e-9, 1), gain=component.vector_gain)
        return loss

    def solve_from_data(
        self,
        solver_data: GlobalIKData,
        problem_data: JaxProblemData,
        model_data: mjx.Data,
    ) -> tuple[GlobalIKSolution, GlobalIKData]:
        """Solve the Global IK problem using pre-computed data.

        :param solver_data: The solver-specific data.
        :param problem_data: The problem-specific data.
        :param model_data: The MuJoCo model data.
        :return: A tuple containing the solver solution and updated solver data.
        """
        return self.solve(model_data.qpos, solver_data=solver_data, problem_data=problem_data)

    def solve(
        self, q: jnp.ndarray, solver_data: GlobalIKData, problem_data: JaxProblemData
    ) -> tuple[GlobalIKSolution, GlobalIKData]:
        """Solve the Global IK problem for a given configuration.

        :param q: The current joint configuration.
        :param solver_data: The solver-specific data.
        :param problem_data: The problem-specific data.
        :return: A tuple containing the solver solution and updated solver data.
        :raises ValueError: If the input configuration has incorrect dimensions.
        """
        if q.shape != (self.model.nq,):
            raise ValueError(f"wrong dimension of the state: expected ({self.model.nq}, ), got {q.shape}")
        grad = self.grad_fn(q, problem_data)

        delta_q, opt_state = self._optimizer.update(grad, solver_data.optax_state)

        return GlobalIKSolution(q_opt=q + delta_q, v_opt=delta_q / self.__dt), GlobalIKData(optax_state=opt_state)

    def init(self, q: mjt.ndarray) -> GlobalIKData:
        """Initialize the Global IK solver data.

        :param q: The initial joint configuration.
        :return: Initialized solver-specific data.
        :raises ValueError: If the input configuration has incorrect dimensions.
        """
        if q.shape != (self.model.nq,):
            raise ValueError(f"Invalid dimension of the velocity: expected ({self.model.nq}, ), got {q.shape}")

        return GlobalIKData(optax_state=self._optimizer.init(q))
