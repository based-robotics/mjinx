import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import optax

import mjinx.typing as mjt
from mjinx import configuration
from mjinx.components.barriers._base import JaxBarrier
from mjinx.components.constraints._base import JaxConstraint
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
    r"""Global Inverse Kinematics solver using gradient-based optimization.

    This solver uses Optax for gradient-based optimization to solve the inverse kinematics
    problem globally. Unlike the local solver, it optimizes over joint positions directly
    rather than velocities, and can potentially find solutions that avoid local minima.

    The optimization problem is formulated as:

    .. math::

        \min_{q} \mathcal{L}(q) = \sum_{i} \|e_i(q)\|^2_{W_i} - \sum_{j} \lambda_j \log(h_j(q))

    where:
        - :math:`e_i(q)` are the task errors (f_i(q) - f_i_desired)
        - :math:`W_i` are the task weight matrices
        - :math:`h_j(q)` are the barrier constraints
        - :math:`\lambda_j` are barrier gains
        - :math:`\log(h_j(q))` is a logarithmic barrier that approaches infinity as :math:`h_j(q)` approaches zero

    The gradient of this loss function is:

    .. math::

        \nabla_q \mathcal{L}(q) = \sum_{i} J_i^T W_i e_i(q) - \sum_{j} \lambda_j \frac{\nabla_q h_j(q)}{h_j(q)}

    where:
        - :math:`J_i` is the Jacobian of task i (∂e_i/∂q)
        - :math:`\nabla_q h_j(q)` is the gradient of barrier j

    The optimization is performed using Optax's gradient-based optimizers, which update
    the joint positions iteratively:

    .. math::

        q_{t+1} = q_t - \alpha_t \nabla_q \mathcal{L}(q_t)

    where :math:`\alpha_t` is the step size determined by the optimizer.

    The logarithmic barrier function ensures that constraints are satisfied by creating
    an infinite penalty as the system approaches constraint boundaries. This allows the
    optimization to explore the full feasible space without explicit inequality constraints.

    Compared to the local IK solver:
    1. Global IK optimizes joint positions directly rather than velocities
    2. It can potentially escape local minima through gradient-based exploration
    3. It uses soft constraints via logarithmic barriers rather than hard inequality constraints
    4. It typically requires more iterations but may find better solutions for complex problems

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
        r"""Compute the logarithmic barrier function.

        The logarithmic barrier function creates a penalty that increases to infinity
        as the constraint value approaches zero:

        .. math::

            B(x) = \lambda \log(x)

        where:
            - :math:`\lambda` is the barrier gain
            - :math:`x` is the constraint value (:math:`h(q)`)

        This function is used to enforce inequality constraints in the optimization,
        creating a continuous penalty that grows rapidly as constraints are approached.

        :param x: The input values (barrier function values).
        :param gain: The gain values for the barrier.
        :return: The computed logarithmic barrier value.
        """
        return jnp.sum(gain * jax.lax.map(jnp.log, x))

    def loss_fn(self, q: jnp.ndarray, problem_data: JaxProblemData) -> float:
        r"""Compute the loss function for the given joint configuration.

        This function evaluates the total loss for the optimization problem:

        .. math::

            \mathcal{L}(q) = \sum_{i} \|e_i(q)\|^2_{W_i} - \sum_{j} \lambda_j \log(h_j(q))

        The first term represents the weighted sum of squared task errors, while the second
        term represents the logarithmic barrier penalties for constraint satisfaction.
        The logarithmic barrier approach transforms inequality constraints into penalty terms,
        creating a barrier that prevents the optimizer from violating the constraints.

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
            if isinstance(component, JaxConstraint):
                # TODO: implement
                raise NotImplementedError("constraints are not supported by Global IK")
        return loss

    def solve_from_data(
        self,
        model_data: mjx.Data,
        solver_data: GlobalIKData,
        problem_data: JaxProblemData,
    ) -> tuple[GlobalIKSolution, GlobalIKData]:
        """Solve the Global IK problem using pre-computed data.

        :param model_data: The MuJoCo model data.
        :param solver_data: The solver-specific data.
        :param problem_data: The problem-specific data.
        :return: A tuple containing the solver solution and updated solver data.
        """
        return self.solve(model_data.qpos, solver_data=solver_data, problem_data=problem_data)

    def solve(
        self, q: jnp.ndarray, solver_data: GlobalIKData, problem_data: JaxProblemData
    ) -> tuple[GlobalIKSolution, GlobalIKData]:
        """Solve the Global IK problem for a given configuration.

        This method performs gradient-based optimization to find a configuration that
        minimizes the task errors while satisfying barrier constraints. Unlike the local
        solver, it operates directly on joint positions rather than velocities.

        :param q: The current joint configuration.
        :param solver_data: The solver-specific data containing optimization state.
        :param problem_data: The problem-specific data containing tasks and barriers.
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
