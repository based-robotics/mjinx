from collections.abc import Sequence
from typing import Generic, TypeVar

import jax.numpy as jnp  # noqa: F401
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
from jaxlie import SE3, SO3

from mjinx.components.constraints._base import Constraint, JaxConstraint
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxEqualityConstraint(JaxConstraint):
    obj1_id: jdc.Static[int]
    obj2_id: jdc.Static[int]
    obj_type: jdc.Static[mj.mjtObj]

    def get_pos1(self, data: mjx.Data) -> jnp.ndarray:
        match self.obj_type:
            case mj.mjtObj.mjOBJ_GEOM:
                return data.geom_xpos[self.obj1_id]
            case mj.mjtObj.mjOBJ_SITE:
                return data.site_xpos[self.obj1_id]
            case _:  # FIXME: what is the best way to handle this?
                return jnp.zeros(3)

    def get_pos2(self, data: mjx.Data) -> jnp.ndarray:
        match self.obj_type:
            case mj.mjtObj.mjOBJ_GEOM:
                return data.geom_xpos[self.obj2_id]
            case mj.mjtObj.mjOBJ_SITE:
                return data.site_xpos[self.obj2_id]
            case _:  # FIXME: what is the best way to handle this?
                return jnp.zeros(3)

    def get_rotation1(self, data: mjx.Data) -> SO3:
        match self.obj_type:
            case mj.mjtObj.mjOBJ_GEOM:
                return SO3.from_matrix(data.geom_xmat[self.obj1_id])
            case mj.mjtObj.mjOBJ_SITE:
                return SO3.from_matrix(data.site_xmat[self.obj1_id])
            case _:  # FIXME: what is the best way to handle this?
                return SO3.identity()

    def get_rotation2(self, data: mjx.Data) -> SO3:
        match self.obj_type:
            case mj.mjtObj.mjOBJ_GEOM:
                return SO3.from_matrix(data.geom_xmat[self.obj2_id])
            case mj.mjtObj.mjOBJ_SITE:
                return SO3.from_matrix(data.site_xmat[self.obj2_id])
            case _:  # FIXME: what is the best way to handle this?
                return SO3.identity()

    def get_frame1(self, data: mjx.Data) -> SE3:
        return SE3.from_rotation_and_translation(self.get_rotation1(data), self.get_pos1(data))

    def get_frame2(self, data: mjx.Data) -> SE3:
        return SE3.from_rotation_and_translation(self.get_rotation2(data), self.get_pos2(data))


AtomicEqualityConstraintType = TypeVar("AtomicEqualityConstraintType", bound=JaxEqualityConstraint)


class EqualityConstraint(Generic[AtomicEqualityConstraintType], Constraint[AtomicEqualityConstraintType]):
    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        obj1_name: str,
        obj2_name: str,
        obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY,
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, gain, mask)
        self._obj1_name = obj1_name
        self._obj2_name = obj2_name
        self._obj_type = obj_type
        self._obj1_id = -1
        self._obj2_id = -1

    @property
    def obj1_name(self) -> str:
        return self._obj1_name

    @property
    def obj2_name(self) -> str:
        return self._obj1_name

    @property
    def obj1_id(self) -> int:
        if self._obj1_id == -1:
            raise ValueError("obj1_id is not available until model is provided.")
        return self._obj1_id

    @property
    def obj2_id(self) -> int:
        if self._obj2_id == -1:
            raise ValueError("obj2_id is not available until model is provided.")
        return self._obj2_id

    @property
    def obj_type(self) -> mj.mjtObj:
        return self._obj_type

    def update_model(self, model: mjx.Model):
        if not self.obj2_name and self.obj_type == mj.mjtObj.mjvBODY:
            raise ValueError("obj2_name is required for bodies.")
        self._obj1_id = mjx.name2id(
            model,
            self._obj_type,
            self._obj1_name,
        )
        self._obj2_id = mjx.name2id(
            model,
            self._obj_type,
            self._obj2_name,
        )
        if self._obj1_id == -1:
            raise ValueError(f"object with type {self._obj_type} and name {self._obj1_name} is not found.")

        if self._obj2_id == -1:
            raise ValueError(f"object with type {self._obj_type} and name {self._obj2_name} is not found.")

        return super().update_model(model)


class ModelEqualityInterface(EqualityConstraint[AtomicEqualityConstraintType]):
    _eq_id: int

    def __init__(self, gain: ArrayOrFloat, model: mjx.Model, eq_name: str = "", eq_id: int = -1):
        if eq_name == "" and eq_id == -1:
            raise ValueError("either eq_name or eq_id must be provided")
        elif eq_name != "":
            self._eq_id = model.name2id(model, mj.mjtObj.mjOBJ_EQUALITY, eq_name)
            if self._eq_id == -1:
                raise ValueError(f"equality constraint {eq_name} not found in model")
        else:
            self._eq_id = eq_id
        obj_type = model.eq_objtype[self._eq_id]

        super().__init__(
            name=model.names[model.name_eqadr[self._eq_id]],
            gain=gain,
            obj1_name=mjx.id2name(model, obj_type, mj.eq_obj1id),
            obj2_name=mjx.id2name(model, obj_type, mj.eq_obj2id),
            obj_type=obj_type,
        )
        self.update_model(model)

    @property
    def eq_id(self) -> int:
        return self._eq_id
