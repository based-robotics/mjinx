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
    limit_type_mask_idxs: jdc.Static[tuple[int, ...]]
    refframe: SE3

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return (self.get_frame(data).inverse() @ self.refframe).log()[self.mask_idxs,]

    @final
    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        T_bt = self.get_frame(data).inverse() @ self.refframe

        def transform_log(tau: jnp.ndarray, frame: SE3) -> jnp.ndarray:
            return (T_bt.multiply(SE3.exp(tau))).log()

        frame_jac = get_frame_jacobian_local(self.model, data, self.obj_id, self.obj_type)

        jlog = jax.jacobian(transform_log)(jnp.zeros(SE3.tangent_dim), T_bt)

        return (-jlog @ frame_jac.T)[self.mask_idxs,]


class FrameBarrier(ObjConstraint[JaxObjConstraint]):
    JaxComponentType: type = JaxObjConstraint
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
        if isinstance(refframe, SE3):
            self._relframe1 = refframe
        elif isinstance(refframe, typing.ndarray):
            self._relframe1 = SE3.from_rotation_and_translation(
                SO3.from_quaternion_xyzw(refframe[[1, 2, 3, 0]]), refframe[:3]
            )
