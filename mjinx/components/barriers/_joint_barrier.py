from typing import Callable, override

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

    qmin: jnp.ndarray
    qmax: jnp.ndarray

    def compute_barrier(self, data: mjx.Data) -> jnp.ndarray:
        # TODO: what the constraint for SO3/SE3 groups is?
        return jnp.concatenate(
            [
                joint_difference(self.model, data.qpos, self.qmin),
                joint_difference(self.model, self.qmax, data.qpos),
            ]
        )


class JointBarrier(Barrier[JaxJointBarrier]):
    __q_min: jnp.ndarray
    __q_max: jnp.ndarray

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
    ):
        super().__init__(name, gain, gain_fn, safe_displacement_gain)
        self.__q_min = self.model.jnt_range[:, 0]
        self.__q_max = self.model.jnt_range[:, 1]

    @property
    def q_min(self) -> jnp.ndarray:
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

    @override
    def _build_component(self) -> JaxJointBarrier:
        return JaxJointBarrier(
            dim=2 * self.model.nv,
            model=self.model,
            gain=self.gain,
            gain_function=self.gain_fn,
            safe_displacement_gain=self.safe_displacement_gain,
            q_min=self.q_min,
            q_max=self.q_max,
        )
