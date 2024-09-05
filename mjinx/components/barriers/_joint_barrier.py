from typing import Callable, Iterable, final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.barriers._base import Barrier, JaxBarrier
from mjinx.configuration import joint_difference
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxJointBarrier(JaxBarrier):
    r"""..."""

    q_min: jnp.ndarray
    q_max: jnp.ndarray

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        # TODO: what the constraint for SO3/SE3 groups is?
        return jnp.concatenate(
            [
                joint_difference(self.model, data.qpos, self.q_min)[self.mask_idxs],
                joint_difference(self.model, self.q_max, data.qpos)[self.mask_idxs],
            ]
        )


class JointBarrier(Barrier[JaxJointBarrier]):
    __q_min: jnp.ndarray | None
    __q_max: jnp.ndarray | None

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        q_min: Iterable | None = None,
        q_max: Iterable | None = None,
        mask: Iterable | None = None,
    ):
        super().__init__(name, gain, gain_fn, safe_displacement_gain, mask=mask)
        if len(self.mask_idxs) != 0:
            self._dim = 2 * self.mask_idxs
        self.__q_min = jnp.array(q_min) if q_min is not None else None
        self.__q_max = jnp.array(q_max) if q_max is not None else None

    @property
    def q_min(self) -> jnp.ndarray:
        if self.__q_min is None:
            raise ValueError(
                "q_min is not yet defined. Either provide it explicitly, or provide an instance of a model"
            )
        return self.__q_min

    @q_min.setter
    def q_min(self, value: np.ndarray | jnp.ndarray):
        self.update_q_min(value)

    def update_q_min(self, q_min: np.ndarray | jnp.ndarray):
        if q_min.shape[-1] != self.model.nv:
            raise ValueError(
                f"[JointBarrier] wrong dimension of q_min: expected {len(self.axes)}, got {q_min.shape[-1]}"
            )

        self._modified = True
        self.__q_min = q_min if isinstance(q_min, jnp.ndarray) else jnp.array(q_min)

    @property
    def q_max(self) -> jnp.ndarray:
        if self.__q_max is None:
            raise ValueError(
                "q_max is not yet defined. Either provide it explicitly, or provide an instance of a model"
            )
        return self.__q_max

    @q_max.setter
    def q_max(self, value: np.ndarray | jnp.ndarray):
        self.update_q_max(value)

    def update_q_max(self, q_max: np.ndarray | jnp.ndarray):
        if q_max.shape[-1] != self.model.nv:
            raise ValueError(
                f"[JointBarrier] wrong dimension of q_max: expected {len(self.axes)}, got {q_max.shape[-1]}"
            )
        self._modified = True
        self.__q_max = q_max if isinstance(q_max, jnp.ndarray) else jnp.array(q_max)

    def update_model(self, model: mjx.Model):
        super().update_model(model)
        self._dim = 2 * self.model.nv
        if self.__q_min is None:
            self.__q_min = self.model.jnt_range[:, 0]
        if self.__q_max is None:
            self.__q_max = self.model.jnt_range[:, 1]
        if self._dim is None:
            self._dim = 2 * self.model.nv

    @final
    def _build_component(self) -> JaxJointBarrier:
        return JaxJointBarrier(
            dim=self.dim,
            model=self.model,
            gain=self.vector_gain,
            gain_function=self.gain_fn,
            safe_displacement_gain=self.safe_displacement_gain,
            q_min=self.q_min,
            q_max=self.q_max,
            mask_idxs=self.mask_idxs,
        )
