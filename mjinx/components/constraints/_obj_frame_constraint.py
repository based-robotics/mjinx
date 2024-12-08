from __future__ import annotations

from collections.abc import Sequence
from typing import final

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
from jaxlie import SE3, SO3

from mjinx import typing
from mjinx.components.constraints._obj_constraint import JaxObjConstraint, ObjConstraint
from mjinx.configuration import get_frame_jacobian_local


@jdc.pytree_dataclass
class JaxFrameConstraint(JaxObjConstraint):
    refframe: SE3

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return (self.get_frame(data).inverse() @ self.refframe).log()[self.mask_idxs,]

    @final
    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        T_bt = self.refframe.inverse() @ self.get_frame(data).inverse()

        def transform_log(tau):
            return (T_bt.multiply(SE3.exp(tau))).log()

        frame_jac = get_frame_jacobian_local(self.model, data, self.obj_id, self.obj_type)
        jlog = jax.jacobian(transform_log)(jnp.zeros(SE3.tangent_dim))
        return (-jlog @ frame_jac.T)[self.mask_idxs,]


class FrameConstraint(ObjConstraint[JaxFrameConstraint]):
    JaxComponentType: type = JaxFrameConstraint
    _refframe: SE3

    def __init__(
        self,
        name: str,
        gain: typing.ArrayOrFloat,
        obj_name: str,
        obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY,
        refframe: typing.ndarray | SE3 | None = None,
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, gain, obj_name, obj_type, mask)
        self.update_refframe(refframe if refframe is not None else SE3.identity())
        self._dim = len(self._mask_idxs) if mask is not None else SE3.tangent_dim

    @property
    def refframe(self) -> SE3:
        return self._refframe

    @refframe.setter
    def refframe(self, value: typing.ArrayOrFloat | SE3):
        self.update_refframe(value)

    def update_refframe(self, refframe: typing.ArrayOrFloat | SE3):
        if not isinstance(refframe, SE3):
            refframe_jnp = jnp.array(refframe)
            if refframe_jnp.shape[-1] != SE3.parameters_dim:
                raise ValueError("target frame provided via array must has length 7 (xyz + quaternion (scalar first))")

            xyz, quat = refframe_jnp[..., :3], refframe_jnp[..., 3:]
            refframe_se3 = SE3.from_rotation_and_translation(
                SO3.from_quaternion_xyzw(
                    quat[..., [1, 2, 3, 0]],
                ),
                xyz,
            )
        else:
            refframe_se3 = refframe
        self._refframe = refframe_se3
