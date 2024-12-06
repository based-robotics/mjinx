from typing import final
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
from jaxlie import SE3

from mjinx import typing
from mjinx.components.constraints._equality._equality_constraint import (
    EqualityConstraint,
    JaxEqualityConstraint,
    ModelEqualityInterface,
)
from mjinx.configuration import get_frame_jacobian_local


@jdc.pytree_dataclass
class JaxJointConstraint(JaxEqualityConstraint):
    polycoefs: jnp.ndarray

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        qposadr1, qposadr2 = self.model.jnt_qposadr[self.obj2_id], self.model.jnt_qposadr[self.obj2_id]
        pos1, pos2 = data.qpos[qposadr1], data.qpos[qposadr1]
        ref1, ref2 = self.model.qpos0[qposadr2], self.model.qpos0[qposadr2]
        dif = pos2 - ref2
        dif_power = jnp.power(dif, jnp.arange(0, 5))
        return pos1 - ref1 - jnp.dot(data[:5], dif_power)

    @final
    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        qposadr2 = self.model.jnt_qposadr[self.obj2_id], self.model.jnt_qposadr[self.obj2_id]
        dofadr1, dofadr2 = self.model.jnt_dofadr[self.obj1_id], self.model.jnt_dofadr[self.obj2_id]
        pos2 = data.qpos[qposadr2]
        ref2 = self.model.qpos0[qposadr2]
        dif = pos2 - ref2
        dif_power = jnp.power(dif, jnp.arange(0, 5))
        deriv = jnp.dot(data[1:5], dif_power[:4] * jnp.arange(1, 5))

        return jnp.zeros(self.model.nv).at[dofadr2].set(-deriv).at[dofadr1].set(1.0)


class JointConstraint(EqualityConstraint[JaxJointConstraint]):
    JaxComponentType: type = JaxJointConstraint
    _polycoefs: jnp.ndarray

    def __init__(
        self,
        name: str,
        gain: typing.ArrayOrFloat,
        obj1_name: str,
        polycoefs: typing.ndarray,
        obj2_name: str = "",
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, gain, obj1_name, obj2_name, obj_type=mj.mjtOBJ.mjtOBJ_JOINT, mask=mask)

        self._dim = len(self._mask_idxs) if mask is not None else 3
        self._polycoefs = jnp.array(polycoefs)

    @property
    def polycoefs(self) -> jnp.ndarray:
        return self._polycoefs


class ModelJointConstraint(ModelEqualityInterface, JointConstraint):
    def __init__(self, gain: typing.ArrayOrFloat, model: mjx.Model, eq_name: str = "", eq_id: int = -1):
        super().__init__(
            gain=gain,
            model=model,
            eq_name=eq_name,
            eq_id=eq_id,
        )
        self._polycoefs = self.model.eq_data[:5]
