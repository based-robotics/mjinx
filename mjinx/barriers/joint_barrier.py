import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from ..configuration import joint_difference
from .base import Barrier


@jdc.pytree_dataclass
class JointBarrier(Barrier):
    r"""..."""

    qmin: jnp.ndarray
    qmax: jnp.ndarray

    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray:
        # TODO: what the constraint for SO3/SE3 groups is?
        return jnp.vstack(
            [
                joint_difference(self.model, data.qpos, -self.qmin),
                joint_difference(self.model, self.qmax, data.qpos),
            ]
        )

    def compute_jacobian(self, data: mjx.Data) -> jax.Array:
        # TODO: handle SO3/SE3 groups
        return super().compute_jacobian(data)
