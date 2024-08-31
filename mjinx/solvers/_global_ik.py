from typing import Callable, override

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import optax

from mjinx import configuration
from mjinx.components.barriers._base import JaxBarrier
from mjinx.components.tasks._base import JaxTask
from mjinx.problem import JaxProblemData
from mjinx.solvers._base import Solver, SolverData


@jdc.pytree_dataclass
class GlobalIKData(SolverData):
    optax_state: optax.OptState


class GlobalIKSolver(Solver[GlobalIKData]):
    _optimizer: optax.GradientTransformation
    __dt: float
    __grad_fn: Callable[[jnp.ndarray, JaxProblemData], float]

    def __init__(self, model: mjx.Model, optimizer: optax.GradientTransformation, dt: float = 1e-2):
        super().__init__(model)
        self._optimizer = optimizer
        self.__grad_fn = jax.grad(
            self.loss_fn,
            argnums=0,
        )
        self.__dt = dt

    def __log_barrier(self, x: jnp.ndarray, gain: jnp.ndarray):
        return jnp.sum(gain * jax.lax.map(jnp.log, x))

    def loss_fn(self, q: jnp.ndarray, problem_data: JaxProblemData) -> float:
        model_data = configuration.update(problem_data.model, q)
        loss = 0

        for component in problem_data.components.values():
            if isinstance(component, JaxTask):
                err = component(model_data)
                loss = loss + component.gain * err.T @ err
            if isinstance(component, JaxBarrier):
                loss = loss - self.__log_barrier(component(model_data), gain=component.gain)
        return loss

    def loss_grad(self, q: jnp.ndarray, problem_data: JaxProblemData) -> jnp.ndarray:
        return self.__grad_fn(q, problem_data)

    @override
    def solve_from_data(
        self,
        problem_data: JaxProblemData,
        model_data: mjx.Data,
        solver_data: GlobalIKData,
    ) -> tuple[jnp.ndarray, GlobalIKData]:
        # print(f"Loss: {GlobalIKSolver.loss_fn(model_data, problem_data)}")
        grad = self.__grad_fn(model_data.qpos, problem_data)
        delta_q, opt_state_raw = self._optimizer.update(grad, solver_data.optax_state)

        return delta_q / self.__dt, GlobalIKData(optax_state=opt_state_raw)

    @override
    def init(self, q: jax.Array) -> GlobalIKData:
        return GlobalIKData(optax_state=self._optimizer.init(q))
