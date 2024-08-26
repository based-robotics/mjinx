"""Build and solve the inverse kinematics problem."""

from functools import partial
from typing import Iterable, Type, TypedDict, Callable

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
import qpax
import jaxopt
from jaxopt import OSQP

from .barriers import Barrier
from .configuration import get_configuration_limit, update
from .tasks import Task


class OSQPParameters(TypedDict):
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
    verbose: bool | int
    implicit_diff: bool
    implicit_diff_solve: Callable | None
    jit: bool
    unroll: jaxopt.base.AutoOrBoolean


class LocalIKSolver:
    def __init__(self, model: mjx.Model, **kwargs: OSQPParameters):
        super().__init__(model)
        self.__osqp_params: OSQPParameters = kwargs

    @property
    def osqp_params(self, OSQPParameters)

    def __compute_qp_objective(
        model: mjx.Model,
        data: mjx.Model,
        tasks: dict[str, Task],
        barriers: dict[str, Barrier],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""..."""

        H = jnp.zeros((model.nv, model.nv))
        c = jnp.zeros(model.nv)
        for task in tasks.values():
            H_task, c_task = task.compute_qp_objective(data)
            H += H_task
            c += c_task

        for barrier in barriers.values():
            H_cbf, c_cbf = barrier.compute_qp_objective(data)
            H += H_cbf
            c += c_cbf

        return (H, c)

    def __compute_qp_inequalities(
        model: mjx.Model,
        data: mjx.Data,
        barriers: dict[str, Barrier],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""..."""
        # TODO: add velocity limits
        G_list = []
        h_list = []

        G_v_limit, h_v_limit = get_configuration_limit(model, 100 * jnp.ones(model.nv))
        G_list.append(G_v_limit)
        h_list.append(h_v_limit)
        for barrier in barriers.values():
            G_barrier, h_barrier = barrier.compute_qp_inequality(data)
            G_list.append(G_barrier)
            h_list.append(h_barrier)

        return jnp.vstack(G_list), jnp.concatenate(h_list)

    def assemble_local_ik(
        model: mjx.Model,
        data: mjx.Data,
        tasks: dict[str, Task],
        barriers: dict[str, Barrier],
        damping: float = 1e-12,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r"""..."""
        P, q = __compute_qp_objective(model, data, tasks, barriers, damping)
        G, h = __compute_qp_inequalities(model, data, barriers)
        return P, q, G, h

    @partial(jax.jit, static_argnames=("maxiter"))
    def solve_local_ik(
        model: mjx.Model,
        q: jnp.ndarray,
        tasks: dict[str, Task],
        barriers: dict[str, Barrier],
        damping: float = 1e-12,
        maxiter: int = 4000,
    ) -> jnp.ndarray:
        r"""..."""
        data = update(model, q)
        P, с, G, h = assemble_local_ik(
            model,
            data,
            tasks,
            barriers,
            damping,
        )
        return OSQP(maxiter=maxiter).run(params_obj=(P, с), params_ineq=(G, h)).params.primal
