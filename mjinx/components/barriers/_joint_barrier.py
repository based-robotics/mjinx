from typing import Callable, Sequence

import itertools
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components._base import Component
from mjinx.components.barriers._base import Barrier, JaxBarrier
from mjinx.configuration import joint_difference, get_joint_zero
from mjinx.typing import ArrayOrFloat, ndarray


@jdc.pytree_dataclass
class JaxJointBarrier(JaxBarrier):
    r"""..."""

    q_min: jnp.ndarray
    q_max: jnp.ndarray
    floating_base: jdc.Static[bool]

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        mask_idxs = tuple(idx + 6 for idx in self.mask_idxs) if self.floating_base else self.mask_idxs
        return jnp.concatenate(
            [
                joint_difference(self.model, data.qpos, self.q_min)[mask_idxs,],
                joint_difference(self.model, self.q_max, data.qpos)[mask_idxs,],
            ]
        )


class JointBarrier(Barrier[JaxJointBarrier]):
    JaxComponentType: type = JaxJointBarrier
    __q_min: jnp.ndarray | None
    __q_max: jnp.ndarray | None

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        q_min: Sequence | None = None,
        q_max: Sequence | None = None,
        mask: Sequence[int] | None = None,
        floating_base: bool = False,
    ):
        super().__init__(name, gain, gain_fn, safe_displacement_gain, mask=mask)
        self.__q_min = jnp.array(q_min) if q_min is not None else None
        self.__q_max = jnp.array(q_max) if q_max is not None else None
        self.__floating_base = floating_base

    @property
    def q_min(self) -> jnp.ndarray:
        if self.__q_min is None:
            raise ValueError(
                "q_min is not yet defined. Either provide it explicitly, or provide an instance of a model"
            )
        return self.__q_min

    @q_min.setter
    def q_min(self, value: ndarray):
        self.update_q_min(value)

    def update_q_min(self, q_min: ndarray):
        if q_min.shape[-1] != self.dim // 2:
            raise ValueError(
                f"[JointBarrier] wrong dimension of q_min: expected {self.dim // 2}, got {q_min.shape[-1]}"
            )

        self.__q_min = get_joint_zero(self.model).at[self.mask_idxs_jnt_space,].set(jnp.array(q_min))

    @property
    def q_max(self) -> jnp.ndarray:
        if self.__q_max is None:
            raise ValueError(
                "q_max is not yet defined. Either provide it explicitly, or provide an instance of a model"
            )
        return self.__q_max

    @q_max.setter
    def q_max(self, value: ndarray):
        self.update_q_max(value)

    def update_q_max(self, q_max: ndarray):
        if q_max.shape[-1] != self.dim // 2:
            raise ValueError(
                f"[JointBarrier] wrong dimension of q_max: expected {self.dim // 2}, got {q_max.shape[-1]}"
            )
        self.__q_max = get_joint_zero(self.model).at[self.mask_idxs_jnt_space,].set(jnp.array(q_max))

    @property
    def mask_idxs_jnt_space(self) -> tuple[int, ...]:
        if self.floating_base:
            return tuple(mask_idx + 7 for mask_idx in self.mask_idxs)
        return self.mask_idxs

    def update_model(self, model: mjx.Model):
        super().update_model(model)

        self._dim = self.model.nv if not self.floating_base else (self.model.nv - 6)
        if len(self.mask_idxs) != self._dim:
            self._dim = 2 * len(self.mask_idxs)
        else:
            self._dim = 2 * self._dim

        begin_idx = 0 if not self.floating_base else 1
        if self.__q_min is None:
            self.q_min = self.model.jnt_range[begin_idx:, 0][self.mask_idxs,]
        if self.__q_max is None:
            self.q_max = self.model.jnt_range[begin_idx:, 1][self.mask_idxs,]

    @property
    def floating_base(self) -> bool:
        return self.__floating_base
