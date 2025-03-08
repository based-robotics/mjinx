"""Build and solve the inverse kinematics problem."""

from collections.abc import Callable
from typing import TypedDict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxopt
import jaxopt.base
import mujoco.mjx as mjx
from jaxopt import OSQP
from typing_extensions import Unpack

import mjinx.typing as mjt
from mjinx.components._base import JaxComponent
from mjinx.components.barriers._base import JaxBarrier
from mjinx.components.tasks._base import JaxTask
from mjinx.problem import JaxProblemData
from mjinx.solvers._base import Solver, SolverData, SolverSolution


class OSQPParameters(TypedDict, total=False):
    """Class which helps to type hint OSQP solver parameters.

    :param check_primal_dual_infeasability: if True populates the ``status`` field of ``state``
        with one of ``BoxOSQP.PRIMAL_INFEASIBLE``, ``BoxOSQP.DUAL_INFEASIBLE``. (default: True).
        If False it improves speed but does not check feasability.
        If jit=False, and if the problem is primal or dual infeasible, then a ValueError exception is raised.
    :param sigma: ridge regularization parameter in linear system.
        Used to stabilize the solution. (default: 1e-6).
    :param momentum: relaxation parameter. (default: 1.6). Must belong to the open interval (0, 2).
        A value of 1 means no relaxation, less than 1 implies under-relaxation, and greater than 1
        implies over-relaxation. Boyd [2, p21] suggests choosing momentum in the range [1.5, 1.8].
    :param eq_qp_solve: method used to solve equality-constrained QP subproblems.
        Options are 'cg', 'cg+jacobi', and 'lu'. (default: 'cg'). 'cg' uses the conjugate gradient method,
        'cg+jacobi' applies Jacobi preconditioning, and 'lu' uses LU factorization for direct solving.
    :param rho_start: initial learning rate for the primal-dual algorithm. (default: 1e-1).
        Determines the step size at the beginning of the optimization process.
    :param rho_min: minimum learning rate for step size adaptation. (default: 1e-6).
        Acts as a lower bound for the step size to prevent excessively small steps.
    :param rho_max: maximum learning rate for step size adaptation. (default: 1e6).
        Acts as an upper bound for the step size to prevent overly large steps.
    :param stepsize_updates_frequency: frequency of stepsize updates during the optimization.
        (default: 10). Every `stepsize_updates_frequency` iterations, the algorithm recomputes the step size.
    :param primal_infeasible_tol: relative tolerance for detecting primal infeasibility. (default: 1e-4).
        Used to declare the problem as infeasible when the primal residual exceeds this tolerance.
    :param dual_infeasible_tol: relative tolerance for detecting dual infeasibility. (default: 1e-4).
        Used to declare the problem as infeasible when the dual residual exceeds this tolerance.
    :param maxiter: maximum number of iterations allowed during optimization. (default: 4000).
        The solver will stop if this iteration count is exceeded.
    :param tol: absolute tolerance for the stopping criterion. (default: 1e-3).
        When the difference in objective values between iterations is smaller than this value, the solver stops.
    :param termination_check_frequency: frequency at which the solver checks for convergence. (default: 5).
        Every `termination_check_frequency` iterations, the solver evaluates if it has converged.
    :param implicit_diff_solve: the solver used to solve linear systems for implicit differentiation.
        Can be any Callable that solves Ax = b, where A is the system matrix and x, b are vectors.

    Note: for the further explanation, see jaxopt.OSQP docstrings
    """

    check_primal_dual_infeasability: jaxopt.base.AutoOrBoolean
    sigma: float
    momentum: float
    eq_qp_solve: str
    rho_start: float
    rho_min: float
    rho_max: float
    stepsize_updates_frequency: int
    primal_infeasible_tol: float
    dual_infeasible_tol: float
    maxiter: int
    tol: float
    termination_check_frequency: int
    implicit_diff_solve: Callable


@jdc.pytree_dataclass
class LocalIKData(SolverData):
    """Data class for the Local Inverse Kinematics solver.

    :param v_prev: The previous velocity solution.
    """

    v_prev: jnp.ndarray


@jdc.pytree_dataclass
class LocalIKSolution(SolverSolution):
    """Solution class for the Local Inverse Kinematics solver.

    :param v_opt: The optimal velocity solution.
    :param dual_var_eq: Dual variables for equality constraints.
    :param dual_var_ineq: Dual variables for inequality constraints.
    :param iter_num: Number of iterations performed.
    :param error: Final error value.
    :param status: Solver status code.
    """

    v_opt: jnp.ndarray
    dual_var_eq: jnp.ndarray
    dual_var_ineq: jnp.ndarray
    iterations: int
    error: float
    status: int


class LocalIKSolver(Solver[LocalIKData, LocalIKSolution]):
    """Local Inverse Kinematics solver using Quadratic Programming.

    This solver uses OSQP to solve a local approximation of the inverse kinematics problem
    as a Quadratic Program.

    :param model: The MuJoCo model.
    :param kwargs: Additional parameters for the OSQP solver.
    """

    def __init__(self, model: mjx.Model, **kwargs: Unpack[OSQPParameters]):
        """Initialize the Local IK solver.

        :param model: The MuJoCo model.
        :param kwargs: Additional parameters for the OSQP solver.
        """
        super().__init__(model)
        self._solver = OSQP(**kwargs)

    def __compute_qp_matrices(
        self,
        problem_data: JaxProblemData,
        model_data: mjx.Data,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute the matrices for the Quadratic Program.

        :param problem_data: The problem-specific data containing model and component information.
        :param model_data: The MuJoCo model data for the current state.
        :return: A tuple containing:
            - P: The quadratic term matrix (H_total in the code)
            - q: The linear term vector (c_total in the code)
            - G: The inequality constraint matrix
            - h: The inequality constraint vector
        """
        nv = problem_data.model.nv

        def process_task(task: JaxTask) -> tuple[jnp.ndarray, jnp.ndarray]:
            """
            Process a task component to compute its contribution to the QP matrices.

            :param task: The task component to process.
            :return: Tuple of (H, c) where H is the quadratic term and c is the linear term.
            """
            jacobian = task.compute_jacobian(model_data)
            minus_gain_error = -task.vector_gain * jax.vmap(task.gain_fn)(task(model_data))  # type: ignore[arg-type]

            weighted_jacobian = task.matrix_cost @ jacobian
            weighted_error = task.matrix_cost @ minus_gain_error

            # Levenberg-Marquardt damping
            mu = task.lm_damping * jnp.dot(weighted_error, weighted_error)
            H = weighted_jacobian.T @ weighted_jacobian + mu * jnp.eye(nv)
            c = -weighted_error.T @ weighted_jacobian
            return H, c

        def process_barrier(barrier: JaxBarrier) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """
            Process a barrier component to compute its contribution to the QP matrices.

            :param barrier: The barrier component to process.
            :return: Tuple of (H, c, G, h) where H and c contribute to the objective,
                    and G and h contribute to the inequality constraints.
            """
            jacobian = barrier.compute_jacobian(model_data)
            gain_over_jacobian = barrier.safe_displacement_gain / (jnp.linalg.norm(jacobian) ** 2)

            # Computing objective term
            H = gain_over_jacobian * jnp.eye(nv)
            c = -gain_over_jacobian * barrier.compute_safe_displacement(model_data)

            # Computing the constraint
            barrier_value = barrier(model_data)
            G = -barrier.compute_jacobian(model_data)
            h = barrier.vector_gain * jax.vmap(barrier.gain_fn)(barrier_value)  # type: ignore[arg-type]

            return H, c, G, h

        H_total = jnp.zeros((self.model.nv, self.model.nv))
        c_total = jnp.zeros(self.model.nv)

        G_list = []
        h_list = []

        # Adding velocity limit
        G_list.append(jnp.eye(problem_data.model.nv))
        G_list.append(-jnp.eye(problem_data.model.nv))
        h_list.append(-problem_data.v_min)
        h_list.append(problem_data.v_max)

        # Process each component
        for component in problem_data.components.values():
            # Tasks
            if isinstance(component, JaxTask):
                H, c = process_task(component)
            # Barriers
            elif isinstance(component, JaxBarrier):
                H, c, G, h = process_barrier(component)
                G_list.append(G)
                h_list.append(h)
            H_total = H_total + H
            c_total = c_total + c

        # Combine all inequality constraints
        return H_total, c_total, jnp.vstack(G_list), jnp.concatenate(h_list)

    def solve_from_data(
        self,
        solver_data: LocalIKData,
        problem_data: JaxProblemData,
        model_data: mjx.Data,
    ) -> tuple[LocalIKSolution, LocalIKData]:
        """Solve the Local IK problem using pre-computed data.

        :param solver_data: The solver-specific data.
        :param problem_data: The problem-specific data.
        :param model_data: The MuJoCo model data.
        :return: A tuple containing the solver solution and updated solver data.
        """
        P, c, G, h = self.__compute_qp_matrices(problem_data, model_data)
        solution = self._solver.run(
            # TODO: warm start is not working
            # init_params=self._solver.init_params(solver_data.v_prev, (P, c), None, (G, h)),
            params_obj=(P, c),
            params_ineq=(G, h),
        )

        return (
            LocalIKSolution(
                v_opt=solution.params.primal,
                dual_var_eq=solution.params.dual_eq,
                dual_var_ineq=solution.params.dual_ineq,
                iterations=solution.state.iter_num,
                error=solution.state.error,
                status=solution.state.status,
            ),
            LocalIKData(v_prev=solution.params.primal),
        )

    def init(self, v_init: mjt.ndarray | None = None) -> LocalIKData:
        """Initialize the Local IK solver data.

        :param v_init: The initial velocity. If None, zero velocity is used.
        :return: Initialized solver-specific data.
        :raises ValueError: If the input velocity has incorrect dimensions.
        """
        v_init_jnp = jnp.array(v_init) if v_init is not None else jnp.zeros(self.model.nv)

        if v_init_jnp.shape != (self.model.nv,):
            raise ValueError(f"Invalid dimension of the velocity: expected ({self.model.nv}, ), got {v_init_jnp.shape}")

        return LocalIKData(v_prev=v_init_jnp)
