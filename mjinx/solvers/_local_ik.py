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
from mjinx.components.barriers._base import JaxBarrier
from mjinx.components.constraints._base import JaxConstraint
from mjinx.components.tasks._base import JaxTask
from mjinx.problem import JaxProblemData
from mjinx.solvers._base import Solver, SolverData, SolverSolution


class OSQPParameters(TypedDict, total=False):
    """Parameters for configuring the OSQP solver.

    :param check_primal_dual_infeasability: If True, populates the ``status`` field of ``state``
        with one of ``BoxOSQP.PRIMAL_INFEASIBLE``, ``BoxOSQP.DUAL_INFEASIBLE`` (default: True).
        If False, improves speed but does not check feasibility.
        When jit=False and the problem is primal or dual infeasible, a ValueError is raised.
    :param sigma: Ridge regularization parameter for stabilizing the linear system solution (default: 1e-6).
    :param momentum: Relaxation parameter (default: 1.6). Must be in the open interval (0, 2).
        A value of 1 means no relaxation, less than 1 means under-relaxation, and greater than 1
        means over-relaxation. For best results, choose momentum in the range [1.5, 1.8].
    :param eq_qp_solve: Method for solving equality-constrained QP subproblems.
        Options are 'cg' (conjugate gradient), 'cg+jacobi' (with Jacobi preconditioning),
        and 'lu' (direct solving via LU factorization). Default is 'cg'.
    :param rho_start: Initial step size for the primal-dual algorithm (default: 1e-1).
    :param rho_min: Minimum allowed step size to prevent excessively small steps (default: 1e-6).
    :param rho_max: Maximum allowed step size to prevent overly large steps (default: 1e6).
    :param stepsize_updates_frequency: Number of iterations between step size adjustments (default: 10).
    :param primal_infeasible_tol: Tolerance for detecting primal infeasibility (default: 1e-4).
    :param dual_infeasible_tol: Tolerance for detecting dual infeasibility (default: 1e-4).
    :param maxiter: Maximum number of iterations allowed before termination (default: 4000).
    :param tol: Absolute tolerance for the stopping criterion (default: 1e-3).
    :param termination_check_frequency: Number of iterations between convergence checks (default: 5).
    :param implicit_diff_solve: Solver for linear systems in implicit differentiation.
        Must be a callable that solves Ax = b for x.
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

    This class extends the base SolverSolution to include additional information
    specific to the Local IK solver's quadratic programming solution.

    :param v_opt: The optimal velocity solution.
    :param dual_var_eq: Dual variables for equality constraints.
    :param dual_var_ineq: Dual variables for inequality constraints.
    :param iterations: Number of iterations performed by the solver.
    :param error: Final error value at convergence.
    :param status: Solver status code indicating outcome (success, infeasible, etc.).
    """

    dual_var_eq: jnp.ndarray
    dual_var_ineq: jnp.ndarray
    iterations: int
    error: float
    status: int


class LocalIKSolver(Solver[LocalIKData, LocalIKSolution]):
    r"""Local Inverse Kinematics solver using Quadratic Programming (QP).

    This solver uses a local linearization approach to solve the inverse kinematics problem.
    At each step, it formulates a Quadratic Program (QP) that approximates the nonlinear optimization
    problem and solves for joint velocities that minimize task errors while respecting constraints.
    
    The QP is formulated as:
    
    .. math::
        
        \min_{v} \frac{1}{2} v^T P v + c^T v \quad \text{subject to} \quad G v \leq h
        
    where:
        - :math:`v` is the vector of joint velocities
        - :math:`P` is a positive-definite matrix constructed from task Jacobians
        - :math:`c` is a vector constructed from task errors
        - :math:`G` and :math:`h` encode barrier constraints and velocity limits
    
    For tasks, the contribution to the objective is:
    
    .. math::
    
        P_{task} &= J^T W J \\
        c_{task} &= -J^T W e
        
    where:
        - :math:`J` is the task Jacobian matrix (∂f/∂q)
        - :math:`W` is the task weight matrix (cost)
        - :math:`e` is the task error (f(q) - f_desired)
    
    For barriers, the constraints are linearized as:
    
    .. math::
    
        G_{barrier} &= -J_b \\
        h_{barrier} &= \alpha h(q)
        
    where:
        - :math:`J_b` is the barrier Jacobian (∂h/∂q)
        - :math:`h(q)` is the barrier function value
        - :math:`\alpha` is a gain parameter that controls constraint relaxation
    
    Additionally, velocity limits are incorporated as:
    
    .. math::
    
        G_{limits} &= \begin{bmatrix} I \\ -I \end{bmatrix} \\
        h_{limits} &= \begin{bmatrix} v_{max} \\ -v_{min} \end{bmatrix}
    
    The solver also includes a safe displacement term to push away from constraint boundaries:
    
    .. math::
    
        P_{safe} &= \beta I \\
        c_{safe} &= -\beta v_{safe}
        
    where :math:`v_{safe}` is a velocity that pushes away from constraint boundaries.
    
    The complete QP matrices are assembled by combining these components:
    
    .. math::
    
        P &= \sum_i P_{task,i} + P_{safe} \\
        c &= \sum_i c_{task,i} + c_{safe} \\
        G &= \begin{bmatrix} G_{barrier} \\ G_{limits} \end{bmatrix} \\
        h &= \begin{bmatrix} h_{barrier} \\ h_{limits} \end{bmatrix}
    
    The QP is solved using the OSQP solver, which implements an efficient primal-dual
    interior point method specifically designed for convex quadratic programs.

    :param model: The MuJoCo model.
    :param dt: The time step for integration.
    :param osqp_params: Parameters for the OSQP solver.
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
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray | None,
        jnp.ndarray | None,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        r"""Compute the matrices for the QP problem.

        This method constructs the matrices needed for the quadratic program:
 
        .. math::
        
            \min_{v} \frac{1}{2} v^T P v + c^T v \quad \text{subject to} \quad G v \leq h
        
        For each task, we add terms to P and c based on:
        
        .. math::
        
            P_{task} &= J^T W J \\
            c_{task} &= -J^T W e
        
        For each barrier, we add constraints to G and h:
        
        .. math::
        
            G_{barrier} &= -J_b \\
            h_{barrier} &= \alpha h(q)
        
        and terms to the objective for safe displacements:
        
        .. math::
        
            P_{safe} &= \beta I \\
            c_{safe} &= -\beta v_{safe}
            
        where :math:`v_{safe}` is a velocity that pushes away from constraint boundaries.

        :param problem_data: The problem-specific data.
        :param model_data: The MuJoCo model data.
        :return: A tuple of (P, c, G, h) matrices for the QP problem.
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

        def process_soft_constraint(constraint: JaxConstraint) -> tuple[jnp.ndarray, jnp.ndarray]:
            jacobian = constraint.compute_jacobian(model_data)
            minus_gain_error = -constraint.vector_gain * jax.vmap(constraint.gain_fn)(constraint(model_data))  # type: ignore[arg-type]

            weighted_jacobian = constraint.soft_constraint_cost @ jacobian
            weighted_error = constraint.soft_constraint_cost @ minus_gain_error

            H = weighted_jacobian.T @ weighted_jacobian
            c = -weighted_error.T @ weighted_jacobian
            return H, c

        def process_hard_constraint(constraint: JaxConstraint) -> tuple[jnp.ndarray, jnp.ndarray]:
            jacobian = constraint.compute_jacobian(model_data)
            bias = -constraint.vector_gain * jax.vmap(constraint.gain_fn)(
                constraint(model_data)  # type: ignore[arg-type]
            )

            return jacobian, bias

        H_total = jnp.zeros((self.model.nv, self.model.nv))
        c_total = jnp.zeros(self.model.nv)

        G_list = []
        h_list = []
        A_list = []
        b_list = []

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
                H_total = H_total + H
                c_total = c_total + c
            # Barriers
            elif isinstance(component, JaxBarrier):
                H, c, G, h = process_barrier(component)
                G_list.append(G)
                h_list.append(h)
                H_total = H_total + H
                c_total = c_total + c
            elif isinstance(component, JaxConstraint):
                if component.hard_constraint:
                    A, b = process_hard_constraint(component)
                    A_list.append(A)
                    b_list.append(b)
                else:
                    H, c = process_soft_constraint(component)
                    H_total = H_total + H
                    c_total = c_total + c

        # Combine all inequality constraints
        return (
            H_total,
            c_total,
            jnp.vstack(A_list) if len(A_list) != 0 else None,
            jnp.concatenate(b_list) if len(b_list) != 0 else None,
            jnp.vstack(G_list),
            jnp.concatenate(h_list),
        )

    def solve_from_data(
        self,
        model_data: mjx.Data,
        solver_data: LocalIKData,
        problem_data: JaxProblemData,
    ) -> tuple[LocalIKSolution, LocalIKData]:
        """Solve the Local IK problem using pre-computed data.

        :param model_data: The MuJoCo model data.
        :param solver_data: The solver-specific data.
        :param problem_data: The problem-specific data.
        :return: A tuple containing the solver solution and updated solver data.
        """
        P, c, A, b, G, h = self.__compute_qp_matrices(problem_data, model_data)
        solution = self._solver.run(
            # TODO: warm start is not working
            # init_params=self._solver.init_params(solver_data.v_prev, (P, c), None, (G, h)),
            params_obj=(P, c),
            params_eq=(A, b) if (A is not None and b is not None) else None,
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
            raise ValueError(
                f"Invalid dimension of the velocity: expected ({self.model.nv}, ), got {v_init_jnp.shape}"
            )

        return LocalIKData(v_prev=v_init_jnp)
