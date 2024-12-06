from collections.abc import Sequence
from typing import final

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
from jaxlie import SE3, SO3

from mjinx import typing
from mjinx.components.constraints._equality._equality_constraint import (
    EqualityConstraint,
    JaxEqualityConstraint,
    ModelEqualityInterface,
)
from mjinx.configuration import get_frame_jacobian_local


@jdc.pytree_dataclass
class JaxWeldConstraint(JaxEqualityConstraint):
    relframe1: SE3
    relframe2: SE3

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        frame1 = self.relframe1 @ self.get_frame1(data)
        frame2 = self.relframe2 @ self.get_frame2(data)
        return (frame1.inverse() @ frame2).log()[self.mask_idxs,]

    @final
    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        T_bt = self.get_frame1(data).inverse() @ self.get_frame2(data).inverse()

        def transform_log(tau: jnp.ndarray, frame: SE3) -> jnp.ndarray:
            return (T_bt.multiply(SE3.exp(tau))).log()

        frame1_jac = get_frame_jacobian_local(self.model, data, self.obj1_id, self.obj_type)
        frame2_jac = get_frame_jacobian_local(self.model, data, self.obj2_id, self.obj_type)

        jlog = jax.jacobian(transform_log)(jnp.zeros(SE3.tangent_dim), T_bt)
        jlog_inv = jax.jacobian(transform_log)(jnp.zeros(SE3.tangent_dim), T_bt.inverse())

        return (jlog @ frame2_jac.T - jlog_inv @ frame1_jac.T)[self.mask_idxs,]


class WeldConstraint(EqualityConstraint[JaxEqualityConstraint]):
    JaxComponentType: type = JaxWeldConstraint
    _relframe1: SE3
    _relframe2: SE3
    # active: bool = True # TODO: what is the best way to implement this?

    def __init__(
        self,
        name: str,
        gain: typing.ArrayOrFloat,
        obj1_name: str,
        obj2_name: str = "",
        obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY,
        relframe1: SE3 | None = None,
        relframe2: SE3 | None = None,
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, gain, obj1_name, obj2_name, obj_type, mask)

        self.update_relframe1(relframe1 if relframe1 is not None else SE3.identity())
        self.update_relframe2(relframe2 if relframe2 is not None else SE3.identity())

        self._dim = len(self._mask_idxs) if mask is not None else SE3.tangent_dim

    @property
    def relframe1(self) -> SE3:
        return self._relframe1

    @relframe1.setter
    def relframe1(self, value: SE3 | typing.ndarray):
        return self.update_relframe1(value)

    @property
    def relframe2(self) -> SE3:
        return self._relframe1

    @relframe2.setter
    def relframe2(self, value: SE3 | typing.ndarray):
        return self.update_relframe2(value)

    def update_relframe1(self, relframe1: SE3 | typing.ndarray):
        if isinstance(relframe1, SE3):
            self._relframe1 = relframe1
        elif isinstance(relframe1, typing.ndarray):
            self._relframe1 = SE3.from_rotation_and_translation(
                SO3.from_quaternion_xyzw(relframe1[[1, 2, 3, 0]]), relframe1[:3]
            )

    def update_relframe2(self, relframe: SE3 | typing.ndarray):
        if isinstance(relframe, SE3):
            self._relframe1 = relframe
        elif isinstance(relframe, typing.ndarray):
            self._relframe1 = SE3.from_rotation_and_translation(
                SO3.from_quaternion_xyzw(relframe[[1, 2, 3, 0]]), relframe[:3]
            )


class ModelWeldConstraint(ModelEqualityInterface, WeldConstraint):
    def __init__(self, gain: typing.ArrayOrFloat, model: mjx.Model, eq_name: str = "", eq_id: int = -1):
        super().__init__(
            gain=gain,
            model=model,
            eq_name=eq_name,
            eq_id=eq_id,
        )
        match self.obj_type:
            case mj.mjtObj.mjOBJ_BODY:
                data = model.eq_data[self.eq_id]
                anchor1, anchor2 = data[0:3], data[3:6]
                relquat = data[6:10]

                # TODO: check if relquat is placed in the correct place
                # and check that reference frames are indeed local
                self.update_relframe1(
                    SE3.from_rotation_and_translation(SO3.from_quaternion_xyzw(relquat[[1, 2, 3, 0]]), anchor1)
                )
                self.update_relframe2(SE3.from_rotation_and_translation(SO3.identity(), anchor2))
            case mj.mjObj.mjOBJ_SITE:
                self.update_relframe1(SE3.identity())
                self.update_relframe2(SE3.identity())
            case _:
                raise ValueError(f"model could contain only body or site weld constraint, got {self.obj_type}")
