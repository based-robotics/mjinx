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

# TODO: maybe passing instance of OSQP is easier to implement, but
# I do not want to directly expose OSQP solver interface to the user (it's little bit ugly)
# plus I want to generalize this for many solvers


class OSQPParameters(TypedDict, total=False):
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
    v_prev: jnp.ndarray


@jdc.pytree_dataclass
class LocalIKSolution(SolverSolution):
    v_opt: jnp.ndarray
    dual_var_eq: jnp.ndarray
    dual_var_ineq: jnp.ndarray
    iter_num: int
    error: float
    status: int


class LocalIKSolver(Solver[LocalIKData, LocalIKSolution]):
    _solver: OSQP

    def __init__(self, model: mjx.Model, **kwargs: Unpack[OSQPParameters]):
        super().__init__(model)
        self._solver = OSQP(**kwargs)

    def __parse_component_objective(
        self,
        model: mjx.Model,
        model_data: mjx.Data,
        component: JaxComponent,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if isinstance(component, JaxTask):
            jacobian = component.compute_jacobian(model_data)
            minus_gain_error = -component.vector_gain * jax.lax.map(component.gain_fn, component(model_data))

            weighted_jacobian = component.matrix_cost @ jacobian  # [cost]
            weighted_error = component.matrix_cost @ minus_gain_error  # [cost]

            mu = component.lm_damping * weighted_error @ weighted_error  # [cost]^2
            # TODO: nv is a dimension of the tangent space, right?..
            eye_tg = jnp.eye(model.nv)
            # Our Levenberg-Marquardt damping `mu * eye_tg` is isotropic in the
            # robot's tangent space. If it helps we can add a tangent-space scaling
            # to damp the floating base differently from joint angular velocities.
            H = weighted_jacobian.T @ weighted_jacobian + mu * eye_tg
            c = -weighted_error.T @ weighted_jacobian

            # TODO: is it possible not to use model from JaxComponent?
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
        r"""..."""
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
        v_init_jnp = jnp.array(v_init) if v_init is not None else jnp.zeros(self.model.nv)

        if v_init_jnp.shape != (self.model.nv,):
            raise ValueError(f"Invalid dimension of the velocity: expected ({self.model.nv}, ), got {v_init_jnp.shape}")

        return LocalIKData(v_prev=v_init_jnp)
