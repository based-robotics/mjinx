"""All barriers derive from the :class:`Barrier` base class.

The formalism used in this implementation is written down in
https://simeon-ned.com/blog/2024/cbf/
"""

from typing import Callable
import abc

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx


@jdc.pytree_dataclass
class Barrier(abc.ABC):
    r"""Abstract base class for barrier.

    A barrier is a function :math:`h(q)` that
    satisfies the following condition:

    .. math::

        \frac{\partial h_j}{\partial q}
        \dot{q} +\alpha_j(h_j(q))
        \geq 0, \quad \forall j

    where :math:`\frac{\partial h_j}{\partial q}`
    are the Jacobians of the constraint functions, :math:`\dot{q}`
    is the joint velocity vector, and :math:`\alpha_j` are extended
    `class kappa <https://en.wikipedia.org/wiki/Class_kappa_function>`__
    functions.

    On top of that, following `this article
    <https://arxiv.org/pdf/2404.12329>`__ barriers utilize safe displacement
    term is added to the cost of the optimization problem:

    .. math::

        \frac{r}{2\|J_h\|^{2}}\|dq-dq_{\text{safe}}(q)\|^{2},

    where :math:`J_h` is the Jacobian of the barrier function, dq is the
    joint displacement vector, and :math:`dq_{\text{safe}}(q)` is the safe
    displacement vector.
    """

    gain: jnp.ndarray
    # TODO: how to handle callable function?
    gain_function: Callable[[float], float]
    safe_displacement_gain: float

    @abc.abstractmethod
    def compute_barrier(self, model: mjx.Model, data: mjx.Data) -> jnp.ndarray: ...

    def compute_jacobian(self, model: mjx.Model, data: mjx.Data) -> jnp.ndarray:
        return jax.jacrev(
            lambda q, model, data: self.compute_barrier(
                model,
                data.replace(qpos=q),
            ),
            argnums=0,
        )(data.qpos, model, data)

    def compute_safe_displacement(self, model: mjx.Model, data: mjx.Data) -> jnp.ndarray:
        r""""""
        return jnp.zeros(model.nv)

    def compute_qp_objective(self, model: mjx.Model, data: mjx.Data) -> tuple[jnp.ndarray, jnp.ndarray]:
        r""""""
        gain_over_jacobian = self.safe_displacement_gain / jnp.linalg.norm(self.compute_jacobian(model, data)) ** 2

        return (
            gain_over_jacobian * jnp.eye(model.nv),
            -gain_over_jacobian * self.compute_safe_displacement(model, data),
        )

    def compute_qp_inequality(
        self,
        model: mjx.Model,
        data: mjx.Data,
        dt: float = 1e-3,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r""""""
        return (
            -self.compute_jacobian(model, data) / dt,
            jnp.array(
                [self.gain[i] * self.gain_function(self.compute_barrier(model, data)[i]) for i in range(self.dim)]
            ),
        )

    def __repr__(self) -> str:
        """"""
        return (
            f"Barrier("
            f"gain={self.gain}, "
            f"safe_displacement_gain={self.safe_displacement_gain})"
            f"dim={self.dim})"
        )
