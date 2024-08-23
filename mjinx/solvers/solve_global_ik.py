from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
import optax

from mjinx import configuration
from mjinx.barriers import Barrier
from mjinx.tasks import Task


def log_barrier(x: jnp.ndarray, gain: jnp.ndarray):
    return jnp.sum(gain * jax.lax.map(jnp.log, x))
    # return jnp.sum(gain * jax.lax.map(lambda z: 1 / (1 + jnp.exp(-20 * z)), x))


@jax.jit
def loss_fn(model: mjx.Model, tasks: dict[str, Task], barriers: dict[str, Barrier], q: jnp.ndarray) -> float:
    data = configuration.update(model, q)

    loss = 0.0
    for task in tasks.values():
        err = task.compute_error(data)
        loss = loss + task.gain * err.T @ err
    for barrier in barriers.values():
        loss = loss - log_barrier(barrier.compute_barrier(data), gain=barrier.gain)
    return loss


loss_grad = jax.jit(jax.grad(loss_fn, argnums=3))


@partial(jax.jit, static_argnames=("loss_grad", "optimizer"))
def global_ik_step(
    model: mjx.Model,
    tasks: dict[str, Task],
    barriers: dict[str, Barrier],
    q: jnp.ndarray,
    state: optax.OptState,
    loss_grad: Callable[[mjx.Model, dict[str, Task], jnp.ndarray], jnp.ndarray],
    optimizer: optax.GradientTransformation,
) -> tuple[jnp.ndarray, optax.OptState]:
    grad = loss_grad(model, tasks, barriers, q)
    updates, opt_state = optimizer.update(grad, state)
    q = optax.apply_updates(q, updates)

    return q, opt_state


@partial(jax.jit, static_argnames=("loss_grad", "optimizer", "N_iters_per_task", "dt"))
def integrate_global_ik(
    model: mjx.Model,
    q0: jnp.ndarray,
    tasks: tuple[dict[str, Task]],
    N_iters_per_task: int,
    optimizer: optax.GradientTransformation,
    loss_grad: Callable[[mjx.Model, dict[str, Task], jnp.ndarray], jnp.ndarray],
    dt: float = 1e-3,
):
    qs = [q0]
    opt_state = optimizer.init(q0)
    for tasks_i in tasks:
        q_i = qs[-1].copy()
        for _ in range(N_iters_per_task):
            q_i, opt_state = global_ik_step(model, tasks_i, q_i, opt_state, loss_grad, optimizer)
        qs.append(q_i)
    return jnp.stack(qs)
