"""Build and solve the inverse kinematics problem."""

from typing import Callable, TypedDict

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
    iter_num: int
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

    def __parse_component_objective(
        self,
        model: mjx.Model,
        model_data: mjx.Data,
        component: JaxComponent,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Parse the objective terms for a given component.

        :param model: The MuJoCo model.
        :param model_data: The MuJoCo model data.
        :param component: The component to parse.
        :return: A tuple containing the quadratic and linear terms of the objective.
        """
        if isinstance(component, JaxTask):
            jacobian = component.compute_jacobian(model_data)
            minus_gain_error = -component.vector_gain * jax.lax.map(component.gain_fn, component(model_data))

            weighted_jacobian = component.matrix_cost @ jacobian
            weighted_error = component.matrix_cost @ minus_gain_error

            mu = component.lm_damping * weighted_error @ weighted_error
            eye_tg = jnp.eye(model.nv)
            H = weighted_jacobian.T @ weighted_jacobian + mu * eye_tg
            c = -weighted_error.T @ weighted_jacobian

            return H, c

        elif isinstance(component, JaxBarrier):
            gain_over_jacobian = (
                component.safe_displacement_gain / jnp.linalg.norm(component.compute_jacobian(model_data)) ** 2
            )

            return (
                gain_over_jacobian * jnp.eye(model.nv),
                -gain_over_jacobian * component.compute_safe_displacement(model_data),
            )
        else:
            return jnp.zeros(model.nv, model.nv), jnp.zeros(model.nv)

    def __parse_component_constraints(
        self,
        model: mjx.Model,
        model_data: mjx.Data,
        component: JaxComponent,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Parse the constraint terms for a given component.

        :param model: The MuJoCo model.
        :param model_data: The MuJoCo model data.
        :param component: The component to parse.
        :return: A tuple containing the constraint matrix and right-hand side.
        """
        if isinstance(component, JaxBarrier):
            barrier = component.compute_barrier(model_data)

            return (
                -component.compute_jacobian(model_data),
                component.vector_gain * jax.lax.map(component.gain_fn, barrier),
            )
        else:
            return jnp.empty((0, model.nq)), jnp.empty(0)

    def __compute_qp_matrices(
        self,
        problem_data: JaxProblemData,
        model_data: mjx.Data,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute the matrices for the Quadratic Program.

        :param problem_data: The problem-specific data.
        :param model_data: The MuJoCo model data.
        :return: A tuple containing the quadratic term, linear term, constraint matrix, and right-hand side.
        """
        H_total = jnp.zeros((self.model.nv, self.model.nv))
        c_total = jnp.zeros(self.model.nv)

        G_list = []
        h_list = []

        # Adding velocity limit
        G_list.append(jnp.eye(problem_data.model.nv))
        G_list.append(-jnp.eye(problem_data.model.nv))
        h_list.append(-problem_data.v_min)
        h_list.append(problem_data.v_max)

        for component in problem_data.components.values():
            # The objective
            H, c = self.__parse_component_objective(problem_data.model, model_data, component)
            H_total += H
            c_total += c

            # The constraints
            G, h = self.__parse_component_constraints(problem_data.model, model_data, component)
            G_list.append(G)
            h_list.append(h)

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
            params_obj=(P, c),
            params_ineq=(G, h),
        )

        return (
            LocalIKSolution(
                v_opt=solution.params.primal,
                dual_var_eq=solution.params.dual_eq,
                dual_var_ineq=solution.params.dual_ineq,
                iter_num=solution.state.iter_num,
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
