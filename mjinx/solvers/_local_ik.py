"""Build and solve the inverse kinematics problem."""

from typing import Callable, NotRequired, TypedDict, override

import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxopt
import mujoco.mjx as mjx
from jaxopt import OSQP
from typing_extensions import Unpack

from mjinx.components._base import JaxComponent
from mjinx.problem import JaxProblemData
from mjinx.solvers._base import Solver, SolverState

# TODO: maybe passing instance of OSQP is easier to implement, but
# I do not want to directly expose OSQP solver interface to the user (it's little bit ugly)
# plus I want to generalize this for many solvers


class OSQPParameters(TypedDict):
    check_primal_dual_infeasability: NotRequired[jaxopt.base.AutoOrBoolean]
    sigma: NotRequired[float]
    momentum: NotRequired[float]
    eq_qp_solve: NotRequired[str]
    rho_start: NotRequired[float]
    rho_min: NotRequired[float]
    rho_max: NotRequired[float]
    stepsize_updates_frequency: NotRequired[int]
    primal_infeasible_tol: NotRequired[float]
    dual_infeasible_tol: NotRequired[float]
    maxiter: NotRequired[int]
    tol: NotRequired[float]
    termination_check_frequency: NotRequired[int]
    implicit_diff_solve: NotRequired[Callable]


@jdc.pytree_dataclass
class LocalIKState(SolverState):
    q_prev: jnp.ndarray | None


class LocalIKSolver(Solver[LocalIKState]):
    _solver: OSQP

    def __init__(self, model: mjx.Model, **kwargs: Unpack[OSQPParameters]):
        super().__init__(model)
        self._solver = OSQP(**kwargs)

    def __compute_qp_objective(
        self,
        data: mjx.Model,
        components: dict[str, JaxComponent],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""..."""

        H = jnp.zeros((self.model.nv, self.model.nv))
        c = jnp.zeros(self.model.nv)
        for component in components.values():
            H, c = component.compute_qp_objective(data)
            H += H
            c += c

        return (H, c)

    def __compute_qp_inequalities(
        self,
        data: mjx.Data,
        components: JaxProblemData,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""..."""
        G_list = []
        h_list = []

        # TODO: fix velocity limit?
        # G_v_limit, h_v_limit = get_configuration_limit(model, 100 * jnp.ones(model.nv))
        # G_list.append(G_v_limit)
        # h_list.append(h_v_limit)
        for component in components.values():
            G_barrier, h_barrier = component.compute_qp_inequality(data)
            G_list.append(G_barrier)
            h_list.append(h_barrier)

        return jnp.vstack(G_list), jnp.concatenate(h_list)

    def assemble_local_ik(
        self, data: mjx.Data, components: JaxProblemData
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r"""..."""
        P, q = self.__compute_qp_objective(data, components)
        G, h = self.__compute_qp_inequalities(data, components)
        return P, q, G, h

    @override
    def solve_from_data(
        self,
        data: mjx.Data,
        solver_state: LocalIKState,
        components: JaxProblemData,
    ) -> tuple[jnp.ndarray, LocalIKState]:
        super().solve_from_data(data, components)
        P, с, G, h = self.assemble_local_ik(data, components)
        solution = self._solver.run(
            init_params=self._solver.init_params(solver_state.q_prev, (P, с), (G, h), None),
            params_obj=(P, с),
            params_ineq=(G, h),
        ).params.primal

        return (update(), LocalIKState(solution))

    @override
    def init(self, data: mjx.Data | jnp.ndarray) -> LocalIKState:
        # TODO: ensure this works
        return LocalIKState(None)
