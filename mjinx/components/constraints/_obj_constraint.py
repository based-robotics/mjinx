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
class JaxObjConstraint(JaxConstraint):
    obj_id: jdc.Static[int]
    obj_type: jdc.Static[mj.mjtObj]

    def get_pos(self, data: mjx.Data) -> jnp.ndarray:
        match self.obj_type:
            case mj.mjtObj.mjOBJ_GEOM:
                return data.geom_xpos[self.obj_id]
            case mj.mjtObj.mjOBJ_SITE:
                return data.site_xpos[self.obj_id]
            case mj.mjtObj.mjOBJ_BODY:
                return data.xpos[self.obj_id]
            case _:
                return jnp.zeros(3)

    def get_rotation(self, data: mjx.Data) -> SO3:
        match self.obj_type:
            case mj.mjtObj.mjOBJ_GEOM:
                return SO3.from_matrix(data.geom_xmat[self.obj_id])
            case mj.mjtObj.mjOBJ_SITE:
                return SO3.from_matrix(data.site_xmat[self.obj_id])
            case mj.mjtObj.mjOBJ_BODY:
                return SO3.from_matrix(data.xmat[self.obj_id])
            case _:
                return SO3.identity()

    def get_frame(self, data: mjx.Data) -> SE3:
        return SE3.from_rotation_and_translation(self.get_rotation(data), self.get_pos(data))


AtomicObjConstraintType = TypeVar("AtomicObjConstraintType", bound=JaxObjConstraint)


class ObjConstraint(Generic[AtomicObjConstraintType], Constraint[AtomicObjConstraintType]):
    _obj_name: str
    _obj_id: int
    _obj_type: mj.mjtObj

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        obj_name: str,
        obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY,
        mask: Sequence[int] | None = None,
    ):
        if obj_type not in {mj.mjtObj.mjOBJ_BODY, mj.mjtObj.mjOBJ_SITE, mj.mjtObj.mjOBJ_GEOM}:
            raise ValueError(
                f"Invalid object type: given {obj_type},"
                f" expected {mj.mjtObj.mjOBJ_BODY, mj.mjtObj.mjOBJ_SITE, mj.mjtObj.mjOBJ_GEOM}"
            )

        super().__init__(name, gain, mask)
        self._obj_name = obj_name
        self._obj_type = obj_type
        self._obj_id = -1

    @property
    def obj_name(self) -> str:
        return self._obj_name

    @property
    def obj_id(self) -> int:
        if self._obj_id == -1:
            raise ValueError("body_id is not available until model is provided.")
        return self._obj_id

    @property
    def obj_type(self) -> mj.mjtObj:
        return self._obj_type

    def update_model(self, model: mjx.Model):
        self._obj_id = mjx.name2id(
            model,
            self._obj_type,
            self._obj_name,
        )
        if self._obj_id == -1:
            raise ValueError(f"object with type {self._obj_type} and name {self._obj_name} is not found.")

        return super().update_model(model)
