"""All barriers derive from the :class:`Barrier` base class.

The formalism used in this implementation is written down in
https://simeon-ned.com/blog/2024/cbf/
"""

import abc

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

    Attributes:
        dim: Dimension of the barrier.
        gain: linear barrier gain.
        gain_function: function, that defines stabilization term as nonlinear
            function of barrier. Defaults to the (linear) identity function.
        safe_displacement: Safe backup displacement.
        safe_displacement_gain: positive gain for safe backup displacement.
    """

    gain: jnp.ndarray
    # TODO: how to handle callable function?
    gain_function: callable[[float], float]
    safe_displacement: jnp.ndarray
    safe_displacement_gain: float

    @abc.abstractmethod
    def compute_barrier(self, model: mjx.Model, data: mjx.Data) -> jnp.ndarray:
        r"""Compute the value of the barrier function.

        The barrier function :math:`h(q)`
        is a vector-valued function that represents the safety constraints.
        It should be designed such that the set
        :math:`\{q : h(q) \geq 0\}`
        represents the safe region of the configuration space.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Value of the barrier function :math:`h(q)`.
        """

    @abc.abstractmethod
    def compute_jacobian(self, model: mjx.Model, data: mjx.Data) -> jnp.ndarray:
        r"""Compute the Jacobian matrix of the barrier function.

        The Jacobian matrix
        :math:`\frac{\partial h}{\partial q}(q)`
        of the barrier function with respect to the configuration variables is
        required for the computation of the barrier condition.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix
            :math:`\frac{\partial h}{\partial q}(q)`.
        """

    def compute_safe_displacement(self, model: mjx.Model, data: mjx.Data) -> jnp.ndarray:
        r"""Compute the safe backup displacement.

        The safe backup control displacement :math:`dq_{safe}(q)`
        is a joint displacement vector that can guarantee that system
        would stay in safety set.

        By default, it is set to zero, since it could not violate safety set.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Safe backup joint velocities
                :math:`\dot{q}_{safe}(q)`.
        """
        return jnp.zeros(model.nv)

    def compute_qp_objective(self, model: mjx.Model, data: mjx.Data) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Compute the quadratic objective function for the barrier-based QP.

        The quadratic objective function includes a regularization term
        based on the safe backup policy:

        .. math::

            \gamma(q)\left\| \dot{q}-
            \dot{q}_{safe}(q)\right\|^{2}

        where :math:`\gamma(q)` is a configuration-dependent weight and
        :math:`\dot{q}_{safe}(q)` is the safe backup policy.

        Note:
            If `safe_displacement_gain` is set to zero, the regularization
            term is not included. Jacobian and barrier values are cached
            to avoid recomputation.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Time step for discrete-time implementation. Defaults to 1e-3.

        Returns:
            Tuple containing the quadratic objective matrix (H) and linear
                objective vector (c).
        """
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
        r"""Compute the linear inequality constraints for the barrier-based QP.

        The linear inequality constraints enforce the barrier conditions:

        .. math::

            \frac{\partial h_j}
            {\partial q} \dot{q} +
            \alpha_j(h_j(q)) \geq 0, \quad \forall j

        where :math:`\frac{\partial h_j}{\partial q}`
        are the Jacobians of the constraint functions,
        :math:`\dot{q}` is the joint velocity vector,
        and :math:`\alpha_j` are extended class K functions.

        Note:
            Jacobian and barrier values are cached to avoid recomputation.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Time step for discrete-time implementation. Defaults to 1e-3.

        Returns:
            Tuple containing the inequality constraint matrix (G)
                and vector (h).
        """
        return (
            -self.compute_jacobian(model, data) / dt,
            jnp.array(
                [self.gain[i] * self.gain_function(self.compute_barrier(model, data)[i]) for i in range(self.dim)]
            ),
        )

    def __repr__(self) -> str:
        """Return a string representation of the barrier.

        Returns:
            str: String representation.
        """
        return (
            f"Barrier("
            f"gain={self.gain}, "
            f"safe_displacement_gain={self.safe_displacement_gain})"
            f"dim={self.dim})"
        )
