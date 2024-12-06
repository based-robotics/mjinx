from collections.abc import Sequence
from typing import final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx import typing
from mjinx.components.constraints._equality._equality_constraint import (
    EqualityConstraint,
    JaxEqualityConstraint,
    ModelEqualityInterface,
)
from mjinx.configuration import get_frame_jacobian_world_aligned


@jdc.pytree_dataclass
class JaxConnectConstraint(JaxEqualityConstraint):
    refpos1: jnp.ndarray
    refpos2: jnp.ndarray

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        p1 = self.get_pos1(data) + self.get_rotation1(data).as_matrix @ self.refpos1
        p2 = self.get_pos2(data) + self.get_rotation2(data).as_matrix @ self.refpos2

        return (p1 - p2)[self.mask_idxs,]

    @final
    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        frame1_jac = get_frame_jacobian_world_aligned(self.model, data, self.obj1_id, self.obj_type)
        frame2_jac = get_frame_jacobian_world_aligned(self.model, data, self.obj2_id, self.obj_type)

        return frame1_jac - frame2_jac


class ConnectConstraint(EqualityConstraint[JaxEqualityConstraint]):
    JaxComponentType: type = JaxConnectConstraint
    _refpos1: jnp.ndarray

    def __init__(
        self,
        name: str,
        gain: typing.ArrayOrFloat,
        obj1_name: str,
        obj2_name: str = "",
        obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY,
        refpos1: jnp.ndarray | None = None,
        refpos2: jnp.ndarray | None = None,
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, gain, obj1_name, obj2_name, obj_type, mask)

        self._refpos1 = jnp.zeros(3) if refpos1 is None else refpos1
        self._refpos2 = jnp.zeros(3) if refpos2 is None else refpos2

        self._dim = len(self._mask_idxs) if mask is not None else 3

    @property
    def refpos1(self) -> jnp.ndarray:
        return self._refpos1

    @refpos1.setter
    def refpos1(self, value: typing.ndarray):
        self._refpos1 = jnp.array(value)

    @property
    def refpos2(self) -> jnp.ndarray:
        return self._refpos2

    @refpos2.setter
    def refpos2(self, value: typing.ndarray):
        self._refpos2 = jnp.array(value)


class ModelConnectConstraint(ModelEqualityInterface, ConnectConstraint):
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
                self.refpos1 = data[0:3]
                self.refpos2 = data[3:6]
            case mj.mjObj.mjOBJ_SITE:
                self.refpos1 = jnp.zeros(3)
                self.refpos2 = jnp.zeros(3)
            case _:
                raise ValueError(f"model could contain only body or site weld constraint, got {self.obj_type}")
