from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
import optax
from jaxlie import SE3, SO3

from mjinx import configuration
from mjinx.tasks import FrameTask, Task


@jax.jit
def loss_fn(model: mjx.Model, tasks: dict[str, Task], q: jnp.ndarray) -> float:
    data = configuration.update(model, q)

    loss = 0.0
    for task in tasks.values():
        err = task.compute_error(data)
        loss = loss + task.gain * err.T @ err

    return loss


loss_grad = jax.jit(jax.grad(loss_fn, argnums=2))


@partial(jax.jit, static_argnames=("loss_grad", "optimizer"))
def global_ik_step(
    model: mjx.Model,
    tasks: dict[str, Task],
    q: jnp.ndarray,
    state: optax.OptState,
    loss_grad: Callable[[mjx.Model, dict[str, Task], jnp.ndarray], jnp.ndarray],
    optimizer: optax.GradientTransformation,
) -> tuple[jnp.ndarray, optax.OptState]:
    grad = loss_grad(model, tasks, q)
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
