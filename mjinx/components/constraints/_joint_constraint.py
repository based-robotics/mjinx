from collections.abc import Sequence

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components.constraints._base import Constraint, JaxConstraint
from mjinx.configuration import get_joint_zero
from mjinx.typing import ArrayOrFloat, ndarray


@jdc.pytree_dataclass
class JaxJointConstraint(JaxConstraint):
    q_fixed: jnp.ndarray
    floating_base: jdc.Static[bool]

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        mask_idxs = tuple(idx + 6 for idx in self.mask_idxs) if self.floating_base else self.mask_idxs
        return jnp.concatenate([(data.qpos - self.q_fixed)[mask_idxs,]])

    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        return (
            jnp.eye(self.dim, self.model.nv, 6)[self.mask_idxs,]
            if self.floating_base
            else jnp.eye(self.model.nv)[self.mask_idxs,]
        )


class JointConstraint(Constraint[JaxJointConstraint]):
    JaxComponentType: type = JaxJointConstraint
    _q_fixed: jnp.ndarray | None

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        q_fixed: Sequence | None = None,
        mask: Sequence[int] | None = None,
        floating_base: bool = False,
    ):
        super().__init__(name, gain, mask=mask)
        self._q_fixed = jnp.array(q_fixed) if q_fixed is not None else None
        self.__floating_base = floating_base

    @property
    def q_fixed(self) -> jnp.ndarray:
        if self._q_fixed is None:
            raise ValueError(
                "q_max is not yet defined. Either provide it explicitly, or provide an instance of a model"
            )
        return self._q_max

    @q_fixed.setter
    def q_fixed(self, value: ndarray):
        self.update_q_fixed(value)

    def update_q_fixed(self, q_fixed: ndarray):
        if q_fixed.shape[-1] != self.dim // 2:
            raise ValueError(
                f"[JointConstraint] wrong dimension of q_fixed: expected {self.dim // 2}, got {q_fixed.shape[-1]}"
            )
        self._q_max = jnp.array(q_fixed)

    @property
    def mask_idxs_jnt_space(self) -> tuple[int, ...]:
        if self.floating_base:
            return tuple(mask_idx + 7 for mask_idx in self.mask_idxs)
        return self.mask_idxs

    def update_model(self, model: mjx.Model):
        super().update_model(model)

        self._dim = self.model.nv if not self.floating_base else self.model.nv - 6
        self._mask = jnp.zeros(self._dim)
        self._mask_idxs = tuple(range(self._dim))

        begin_idx = 0 if not self.floating_base else 1
        if self._q_fixed is None:
            self.q_max = self.model.jnt_range[begin_idx:, 1][self.mask_idxs,]

    @property
    def full_q_fixed(self) -> jnp.ndarray:
        if self._model is None:
            raise ValueError("model is not defined yet.")
        return get_joint_zero(self.model).at[self.mask_idxs_jnt_space,].set(jnp.array(self._q_fixed))

    @property
    def floating_base(self) -> bool:
        return self.__floating_base
