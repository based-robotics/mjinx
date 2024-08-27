"""Center of mass task implementation."""

import warnings
from typing import Callable, final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from typing_extensions import override

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.configuration import get_joint_zero, joint_difference
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxJointTask(JaxTask):
    target_q: jnp.ndarray
    mask: jdc.Static[tuple[int, ...]]

    @final
    @override
    def compute_error(self, data: mjx.Data) -> jnp.ndarray:
        r"""..."""
        return joint_difference(self.model, data.qpos, self.target_q)[self.mask]

    # TODO: jacobian of joint task


class JointTask(Task[JaxJointTask]):
    __target_q: jnp.ndarray
    __joints_mask: tuple[int, ...]

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        frame_name: str,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        include_joints: tuple[str, ...] | tuple[int, ...] = (),
        exclude_joints: tuple[str, ...] | tuple[int, ...] = (),
    ):
        super().__init__(name, cost, gain, frame_name, gain_fn, lm_damping)

        self.__joints_mask = self.__generate_mask(include_joints, exclude_joints)

    def __generate_mask(
        self,
        include_joints: tuple[str, ...] | tuple[int, ...] = (),
        exclude_joints: tuple[str, ...] | tuple[int, ...] = (),
    ) -> tuple[int, ...]:
        if len(include_joints) != 0 and len(include_joints) != 0:
            warnings.warn(
                "[JointTask] joints to exclude are provided, include joints would be ignored",
                stacklevel=2,
            )

        idxs: set[int]
        if len(exclude_joints) != 0:
            idxs = set(range(self.model.nv))
            for jnt in exclude_joints:
                jnt_idx = (
                    jnt
                    if isinstance(jnt, int)
                    else mjx.name2id(
                        self.model,
                        mj.mjtObj.mjOBJ_JOINT,
                        jnt,
                    )
                )
                if jnt_idx == -1:
                    warnings.warn(f"[JointTask] joint {jnt} not found, skipping", stacklevel=2)
                else:
                    idxs.difference_update({jnt_idx})
        elif len(include_joints) != 0:
            for jnt in include_joints:
                jnt_idx = (
                    jnt
                    if isinstance(jnt, int)
                    else mjx.name2id(
                        self.model,
                        mj.mjtObj.mjOBJ_JOINT,
                        jnt,
                    )
                )
                if jnt_idx == -1:
                    warnings.warn(f"[JointTask] joint {jnt} not found, skipping", stacklevel=2)
                else:
                    idxs.add(jnt_idx)
        else:
            idxs = set(range(self.model.nv))

        return tuple(1 for i in range(self.model.nv) if i in idxs)

    @override
    def update_model(self, model: mjx.Model):
        super().update_model(model)
        self.target_q = get_joint_zero(model)

    @property
    def joints_mask(self) -> np.ndarray:
        return np.ndarray(self.__joints_mask)

    @property
    def target_q(self) -> jnp.ndarray:
        return self.__target_q

    @target_q.setter
    def target_q(self, value: jnp.ndarray | np.ndarray):
        self.update_target_q(value)

    def update_target_q(self, target_q: jnp.ndarray | np.ndarray):
        if len(target_q) != len(self.__joints_mask):
            raise ValueError(
                "invalid dimension of the target joints value: "
                f"{len(target_q)} given, expected {len(self.__task_axes_idx)} "
            )
        self._modified = True
        self.__target_q = target_q if isinstance(target_q, jnp.ndarray) else jnp.array(target_q)

    @override
    def _build_component(self) -> JaxJointTask:
        return JaxJointTask(
            dim=len(self.task_axes),
            model=self.model,
            cost=self.cost,
            gain=self.gain,
            gain_function=self.gain_fn,
            lm_damping=self.lm_damping,
            target_q=self.target_q,
            mask=self.__joints_mask,
        )
