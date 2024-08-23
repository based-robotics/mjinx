from dataclasses import field

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
    gain: jnp.ndarray = field(init=False)
    joints_gain: jnp.ndarray

    def __post_init__(self):
        object.__setattr__(self, "dim", self.model.nv)
        object.__setattr__(self, "gain", self.joints_gain)
        # FIXME: why self.gain is object type, when vmap is applied
        # object.__setattr__(self, "gain", jnp.tile(self.gain, 2))

    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray:
        # TODO: what the constraint for SO3/SE3 groups is?
        return jnp.concatenate(
            [
                joint_difference(self.model, data.qpos, self.qmin),
                joint_difference(self.model, self.qmax, data.qpos),
            ]
        )


@jdc.pytree_dataclass
class ModelJointBarrier(JointBarrier):
    qmin: jnp.ndarray = field(init=False)
    qmax: jnp.ndarray = field(init=False)

    def __post_init__(self):
        # FIXME: why self.model.jnt_range is object type, when vmap is applied?
        object.__setattr__(self, "qmin", self.model.jnt_range[:, 0])
        object.__setattr__(self, "qmax", self.model.jnt_range[:, 1])

        super().__post_init__()
