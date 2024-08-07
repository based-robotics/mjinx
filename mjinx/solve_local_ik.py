"""Build and solve the inverse kinematics problem."""

from functools import partial
from typing import Iterable, Type

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
import qpax
from jaxopt import OSQP

from .barriers import Barrier
from .configuration import get_configuration_limit, update
from .tasks import Task


def __compute_qp_objective(
    model: mjx.Model,
    data: mjx.Model,
    tasks: dict[str, Task],
    barriers: dict[str, Barrier],
    damping: float,
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
    return P, q, jnp.array([]).reshape(0, model.nv), jnp.array([]), G, h


@partial(jax.jit, static_argnames=("solver"))
def solve_local_ik(
    model: mjx.Model,
    q: jnp.ndarray,
    tasks: dict[str, Task],
    barriers: dict[str, Barrier],
    damping: float = 1e-12,
    solver: str = "qpax",
) -> jnp.ndarray:
    r"""..."""
    data = update(model, q)
    if solver.lower() == "osqp":
        P, с, A, b, G, h = assemble_local_ik(
            model,
            data,
            tasks,
            barriers,
            damping,
        )
        return OSQP().run(params_obj=(P, с), params_ineq=(G, h)).params.primal
    elif solver.lower() == "qpax":
        return qpax.solve_qp(
            *assemble_local_ik(
                model,
                data,
                tasks,
                barriers,
                damping,
            )
        )[0]


@partial(jax.jit, static_argnames=("dt"))
def integrate_local_ik(
    model: mjx.Model,
    q0: jnp.ndarray,
    tasks: tuple[dict[str, Task]],
    barriers: tuple[dict[str, Barrier]],
    dt: float = 1e-3,
    damping: float = 1e-12,
) -> jnp.ndarray:
    r"""..."""
    N_iters = len(tasks)
    qs = [q0]
    for i in range(N_iters):
        v = solve_local_ik(model, qs[-1], tasks[i], barriers[i], damping)
        qs.append(qs[-1] + v * dt)

    return jnp.stack(qs)
