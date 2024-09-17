from typing import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import optax

import mjinx.typing as mjt
from mjinx import configuration
from mjinx.components.barriers._base import JaxBarrier
from mjinx.components.tasks._base import JaxTask
from mjinx.problem import JaxProblemData
from mjinx.solvers._base import Solver, SolverData, SolverSolution


@jdc.pytree_dataclass
class GlobalIKData(SolverData):
    optax_state: optax.OptState


@jdc.pytree_dataclass
class GlobalIKSolution(SolverSolution):
    q_opt: jnp.ndarray


class GlobalIKSolver(Solver[GlobalIKData, GlobalIKSolution]):
    _optimizer: optax.GradientTransformation
    __dt: float
    __grad_fn: Callable[[jnp.ndarray, JaxProblemData], float]

    def __init__(self, model: mjx.Model, optimizer: optax.GradientTransformation, dt: float = 1e-2):
        super().__init__(model)
        self._optimizer = optimizer
        self.grad_fn = jax.grad(
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

    def solve_from_data(
        self,
        solver_data: GlobalIKData,
        problem_data: JaxProblemData,
        model_data: mjx.Data,
    ) -> tuple[GlobalIKSolution, GlobalIKData]:
        return self.solve(model_data.qpos, solver_data=solver_data, problem_data=problem_data)

    def solve(
        self, q: jnp.ndarray, solver_data: GlobalIKData, problem_data: JaxProblemData
    ) -> tuple[GlobalIKSolution, GlobalIKData]:
        if q.shape != (self.model.nq,):
            raise ValueError(f"wrong dimension of the state: expected ({self.model.nq}, ), got {q.shape}")
        grad = self.grad_fn(q, problem_data)

        delta_q, opt_state = self._optimizer.update(grad, solver_data.optax_state)

        return GlobalIKSolution(q_opt=q + delta_q, v_opt=delta_q / self.__dt), GlobalIKData(optax_state=opt_state)

    def init(self, q: mjt.ndarray) -> GlobalIKData:
        if q.shape != (self.model.nq,):
            raise ValueError(f"Invalid dimension of the velocity: expected ({self.model.nq}, ), got {q.shape}")

        return GlobalIKData(optax_state=self._optimizer.init(q))
