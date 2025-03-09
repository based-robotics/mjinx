"""Build and solve the inverse kinematics problem."""

from collections.abc import Callable
from typing import TypedDict, Optional, Tuple

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
    :param use_analytical_solver: Whether to use analytical solutions for problems with only equality
        constraints or no constraints (default: True).
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
    use_analytical_solver: bool


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

    v_opt: jnp.ndarray
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
    
    For problems with only velocity limits (no other inequality constraints), analytical solutions
    can be used for improved performance:
    
    1. For problems with no equality constraints:
    
    .. math::
    
        v = -P^{-1}c
        
    2. For problems with equality constraints:
    
    .. math::
    
        \begin{bmatrix} P & E^T \\ E & 0 \end{bmatrix} \begin{bmatrix} v \\ \lambda \end{bmatrix} = \begin{bmatrix} -c \\ d \end{bmatrix}
    
    In both cases, the solution is then clipped to satisfy velocity limits:
    
    .. math::
    
        v_{clipped} = \text{clip}(v, v_{min}, v_{max})

    This approach of solving analytically and then clipping to satisfy velocity limits is much faster
    than solving the full QP problem, while still providing good results in practice. For problems
    with other inequality constraints (barriers), the solver falls back to the OSQP solver.

    :param model: The MuJoCo model.
    :param use_analytical_solver: Whether to use analytical solutions for problems with only velocity
        limits (default: True).
    :param osqp_params: Parameters for the OSQP solver.
    """

    def __init__(self, model: mjx.Model, use_analytical_solver: bool = True, **kwargs: Unpack[OSQPParameters]):
        """Initialize the Local IK solver.

        :param model: The MuJoCo model.
        :param use_analytical_solver: Whether to use analytical solutions for problems with only equality
            constraints or no constraints (default: True).
        :param kwargs: Additional parameters for the OSQP solver.
        """
        super().__init__(model)
        self._solver = OSQP(**kwargs)
        self._use_analytical_solver = use_analytical_solver

    def __compute_qp_matrices(
        self,
        problem_data: JaxProblemData,
        model_data: mjx.Data,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """Compute the matrices for the QP problem.

        This method constructs the matrices needed for the quadratic program:
        
        .. math::
        
            \min_{v} \frac{1}{2} v^T P v + c^T v \quad \text{subject to} \quad G v \leq h, E v = d
        
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
        :return: A tuple of (P, c, G, h, E, d) matrices for the QP problem.
                 E and d are None if there are no equality constraints.
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

        # For equality constraints (currently none in the implementation)
        E_list = []
        d_list = []

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
        G = jnp.vstack(G_list) if G_list else jnp.zeros((0, self.model.nv))
        h = jnp.concatenate(h_list) if h_list else jnp.zeros(0)

        # Combine all equality constraints (if any)
        E = jnp.vstack(E_list) if E_list else None
        d = jnp.concatenate(d_list) if d_list else None

        return H_total, c_total, G, h, E, d

    def __has_only_velocity_limits(self, G: jnp.ndarray, h: jnp.ndarray) -> bool:
        """Check if the problem has only velocity limits as inequality constraints.

        :param G: The inequality constraint matrix.
        :param h: The inequality constraint vector.
        :return: True if the problem has only velocity limits, False otherwise.
        """
        # Check if G has exactly 2*nv rows (velocity limits only)
        return G.shape[0] == 2 * self.model.nv

    def __solve_unconstrained(self, P: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        """Solve an unconstrained QP problem analytically.

        For problems with no constraints, the solution is:
        v = -P^{-1}c

        :param P: The quadratic cost matrix.
        :param c: The linear cost vector.
        :return: The optimal velocity.
        """
        # Solve the linear system P*v = -c using Cholesky decomposition
        cfac = jax.scipy.linalg.cho_factor(P)
        return jax.scipy.linalg.cho_solve(cfac, -c)

    def __solve_equality_constrained(
        self, P: jnp.ndarray, c: jnp.ndarray, E: jnp.ndarray, d: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Solve an equality-constrained QP problem analytically using the KKT system.

        For problems with only equality constraints, the solution is given by:

        [P  E^T] [v    ] = [-c]
        [E  0  ] [lambda] = [d ]

        :param P: The quadratic cost matrix.
        :param c: The linear cost vector.
        :param E: The equality constraint matrix.
        :param d: The equality constraint vector.
        :return: A tuple of (v, lambda) where v is the optimal velocity and lambda are the Lagrange multipliers.
        """
        n = P.shape[0]
        m = E.shape[0]

        # Construct the KKT matrix
        kkt_matrix = jnp.block([[P, E.T], [E, jnp.zeros((m, m))]])

        # Construct the right-hand side
        rhs = jnp.concatenate([-c, d])

        # Solve the KKT system using LU decomposition
        lu, piv = jax.scipy.linalg.lu_factor(kkt_matrix)
        solution = jax.scipy.linalg.lu_solve((lu, piv), rhs)
        # Extract the primal and dual variables
        v = solution[:n]
        lambda_eq = solution[n:]

        return v, lambda_eq

    def solve_from_data(
        self,
        solver_data: LocalIKData,
        problem_data: JaxProblemData,
        model_data: mjx.Data,
    ) -> tuple[LocalIKSolution, LocalIKData]:
        """Solve the Local IK problem using pre-computed data.

        This method first checks if analytical solutions can be used:
        1. If there are no constraints (or only velocity limits), it uses the analytical solution
           and clips the result to satisfy velocity limits
        2. Otherwise, it falls back to the OSQP solver for problems with inequality constraints

        :param solver_data: The solver-specific data.
        :param problem_data: The problem-specific data.
        :param model_data: The MuJoCo model data.
        :return: A tuple containing the solver solution and updated solver data.
        """
        P, c, G, h, E, d = self.__compute_qp_matrices(problem_data, model_data)

        # Check if we can use analytical solutions (only when there are no inequality constraints except velocity limits)
        if self._use_analytical_solver and self.__has_only_velocity_limits(G, h):
            # Solve analytically based on whether we have equality constraints
            if E is not None:
                # Case with equality constraints: solve KKT system
                v, lambda_eq = self.__solve_equality_constrained(P, c, E, d)
            else:
                # Case without equality constraints: direct solution
                v = self.__solve_unconstrained(P, c)
                lambda_eq = jnp.zeros(0)

            # Clip the solution to satisfy velocity limits
            v_clipped = jnp.clip(v, problem_data.v_min, problem_data.v_max)

            # Return the clipped solution
            return (
                LocalIKSolution(
                    v_opt=v_clipped,
                    dual_var_eq=lambda_eq,
                    dual_var_ineq=jnp.zeros(G.shape[0]),
                    iterations=1,
                    error=0.0,
                    status=0,  # Success
                ),
                LocalIKData(v_prev=v_clipped),
            )

        # Fall back to OSQP for general case with inequality constraints
        return self.__solve_with_osqp(P, c, G, h, solver_data)

    def __solve_with_osqp(
        self, P: jnp.ndarray, c: jnp.ndarray, G: jnp.ndarray, h: jnp.ndarray, solver_data: LocalIKData
    ) -> tuple[LocalIKSolution, LocalIKData]:
        """Solve the QP problem using the OSQP solver.

        :param P: The quadratic cost matrix.
        :param c: The linear cost vector.
        :param G: The inequality constraint matrix.
        :param h: The inequality constraint vector.
        :param solver_data: The solver-specific data.
        :return: A tuple containing the solver solution and updated solver data.
        """
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
