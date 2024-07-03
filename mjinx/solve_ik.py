"""Build and solve the inverse kinematics problem."""

from typing import Iterable

import jax.numpy as jnp
import mujoco.mjx as mjx
import qpax

from .barriers import Barrier
from .tasks import Task


def __compute_qp_objective(
    model: mjx.Model,
    data: mjx.Model,
    tasks: Iterable[Task],
    barriers: Iterable[Barrier],
    damping: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""..."""

    H = jnp.zeros((model.nv, model.nv))
    c = jnp.zeros(model.nv)
    for task in tasks:
        H_task, c_task = task.compute_qp_objective(model, data)
        H += H_task
        c += c_task

    for barrier in barriers:
        H_cbf, c_cbf = barrier.compute_qp_objective(model, data)
        H += H_cbf
        c += c_cbf

    return (H, c)


def __compute_qp_inequalities(
    model: mjx.Model,
    data: mjx.Data,
    barriers: Iterable[Barrier],
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""..."""

    G_list = []
    h_list = []
    # TODO: add velocity limits
    # for limit in (configuration_limit, velocity_limit):
    #     matvec = limit.compute_qp_inequalities(q, dt)
    #     if matvec is not None:
    #         G_list.append(matvec[0])
    #         h_list.append(matvec[1])
    for barrier in barriers:
        G_barrier, h_barrier = barrier.compute_qp_inequality(model, data, dt)
        G_list.append(G_barrier)
        h_list.append(h_barrier)

    return jnp.vstack(G_list), jnp.hstack(h_list)


def assemble_ik(
    model: mjx.Model,
    data: mjx.Data,
    tasks: Iterable[Task],
    barriers: Iterable[Barrier],
    dt: float,
    damping: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""..."""

    P, q = __compute_qp_objective(model, data, tasks, barriers, damping)
    G, h = __compute_qp_inequalities(model, data, barriers, dt)
    return P, q, jnp.array(), jnp.array(), G, h


def solve_ik(
    model: mjx.Model,
    data: mjx.Data,
    tasks: Iterable[Task],
    barriers: Iterable[Barrier],
    dt: float,
    solver: str,
    damping: float = 1e-12,
    **kwargs,
) -> jnp.ndarray:
    r"""..."""

    return qpax.solve_qp(assemble_ik(model, data, tasks, barriers, dt, damping))[0] / dt
