"""Frame task implementation."""

from typing import Callable, Iterable, final

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3

from mjinx.components.tasks._body_task import BodyTask, JaxBodyTask
from mjinx.configuration import get_frame_jacobian_local, get_transform_frame_to_world
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxFrameTask(JaxBodyTask):
    r""""""

    target_frame: SE3

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        r""""""
        return (
            get_transform_frame_to_world(
                self.model,
                data,
                self.body_id,
            ).inverse()
            @ self.target_frame
        ).log()[self.mask_idxs]

    @final
    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        T_bt = self.target_frame.inverse() @ get_transform_frame_to_world(
            self.model,
            data,
            self.body_id,
        )

        def transform_log(tau):
            return (T_bt.multiply(SE3.exp(tau))).log()

        frame_jac = get_frame_jacobian_local(self.model, data, self.body_id)
        jlog = jax.jacobian(transform_log)(jnp.zeros(self.dim))

        # TODO: is indexing correct
        return (-jlog @ frame_jac.T)[self.mask_idxs]


class FrameTask(BodyTask[JaxFrameTask]):
    __target_frame: SE3

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        body_name: str,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Sequence | None = None,
    ):
        super().__init__(name, cost, gain, body_name, gain_fn, lm_damping, mask)
        self.target_frame = SE3.identity()
        self._dim = SE3.tangent_dim if mask is None else len(self.mask_idxs)

    @property
    def target_frame(self) -> SE3:
        return self.__target_frame

    @target_frame.setter
    def target_frame(self, value: SE3 | jnp.ndarray | np.ndarray):
        self.update_target_frame(value)

    def update_target_frame(self, target_frame: SE3 | jnp.ndarray | np.ndarray):
        self._modified = True
        if not isinstance(target_frame, SE3):
            if target_frame.shape[-1] != SE3.parameters_dim:
                raise ValueError("target frame provided via array must has length 7 (xyz + quaternion (scalar first))")

            target_frame = jnp.array(target_frame)
            xyz, quat = target_frame[..., :3], target_frame[..., 3:]
            target_frame = SE3.from_rotation_and_translation(
                SO3.from_quaternion_xyzw(
                    quat[..., [1, 2, 3, 0]],
                ),
                xyz,
            )

        self.__target_frame = target_frame

    @final
    def _build_component(self) -> JaxFrameTask:
        return JaxFrameTask(
            dim=self.dim,
            model=self.model,
            cost=self.matrix_cost,
            gain=self.vector_gain,
            gain_function=self.gain_fn,
            lm_damping=self.lm_damping,
            body_id=self.body_id,
            target_frame=self.target_frame,
            mask_idxs=self.mask_idxs,
        )
