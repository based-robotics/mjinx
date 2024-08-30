from typing import override

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import optax

from mjinx.components._base import JaxComponent
from mjinx.components.barriers._base import JaxBarrier
from mjinx.components.tasks._base import JaxTask
from mjinx.problem import JaxProblemData
from mjinx.solvers._base import Solver, SolverData


@jdc.pytree_dataclass
class GlobalIKState(SolverData):
    optax_state: optax.OptState


class GlobalIKSolver(Solver[GlobalIKState]):
    _optimizer: optax.GradientTransformation

    def __init__(self, model: mjx.Model, optimiezr: optax.GradientTransformation):
        super().__init__(model)
        self._optimizer = optimiezr

    def __log_barrier(x: jnp.ndarray, gain: jnp.ndarray):
        return jnp.sum(gain * jax.lax.map(jnp.log, x))

    def loss_fn(self, data: mjx.Data, components: dict[str, JaxComponent]) -> float:
        loss = 0.0
        for component in components.values():
            # TODO: how to avoid this?..
            match type(component):
                case JaxTask():
                    err = component(data)
                    loss = loss + component.gain * err.T @ err
                case JaxBarrier():
                    loss = loss - self.__log_barrier(component(data), gain=component.gain)
        return loss

    loss_grad = jax.grad(loss_fn, argnums=3)

    @override
    def solve_from_data(
        self,
        data: mjx.Data,
        solver_state: GlobalIKState,
        componets: dict[str, JaxComponent],
        damping: float,
    ) -> tuple[jnp.ndarray, GlobalIKState]:
        grad = self.loss_grad(data, componets)
        updates, opt_state = self._optimizer.update(grad, solver_state.optax_state)
        # q = optax.apply_updates(q, updates)
        # FIXME: updates is in fact just a delta q: the updates function just adds it to the state.
        # But how should we treat it though?..
        return updates, opt_state

    @override
    def init(self, data: mjx.Data | jax.Array) -> GlobalIKState:
        return GlobalIKState(self._optimizer.init(data.qpos))
