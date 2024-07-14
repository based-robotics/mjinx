"""Center of mass task implementation."""

from dataclasses import field

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
from typing_extensions import override

from ..configuration import get_joint_zero, joint_difference
from .base import Task


@jdc.pytree_dataclass
class JointTask(Task):
    target_q: jnp.ndarray
    mask: jnp.ndarray

    def __post_init__(self):
        object.__setattr__(self, "dim", self.model.nv)

    @override
    def compute_error(self, data: mjx.Data) -> jnp.ndarray:
        r"""..."""
        return self.mask * joint_difference(self.model, data.qpos, self.target_q)

    @override
    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        pass


@jdc.pytree_dataclass
class RegularizationTask(JointTask):
    target_q: jnp.ndarray = field(init=False, repr=False)
    mask: jnp.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "target_q", False)
        object.__setattr__(self, "mask", get_joint_zero(self.model))

        super().__post_init__()
