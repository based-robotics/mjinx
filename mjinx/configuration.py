#!/usr/bin/env python3

"""..."""

import jax.numpy as jnp
import numpy as np
import jax
from mujoco import mjx


class Configuration:
    """"""

    def __init__(
        self,
        model: mjx.Model,
    ):
        """..."""
        self.__model = model

    def update(self, q: jnp.ndarray) -> mjx.Data:
        """..."""
        data = mjx.make_data(self.__model)
        data.replace(qpos=q)
        data = mjx.kinematics(self.__model, data)
        data = mjx.com_pos(self.__model, data)

        return data

    def check_limits(self, data: mjx.Data) -> jnp.bool:
        """..."""
        return jnp.all(self.__model.jnt_range[:, 0] < data.qpos < self.__model.jnt_range[:, 1])

    def get_frame_jacobian(self, data: mjx.Data, body_id: int) -> jnp.array:
        """Compute pair of (NV, 3) Jacobians of global point attached to body."""

        def fn(carry, b):
            return b if carry is None else b + carry

        mask = (jnp.arange(self.__model.nbody) == body_id) * 1
        # Puts 1 for all parent links of specified body.
        mask = mjx._src.scan.body_tree(self.__model, fn, "b", "b", mask, reverse=True)
        # From all parent links, select only those which add degree of freedoms?..
        mask = mask[jnp.array(self.__model.dof_bodyid)] > 0

        # Subtree_com is the center of mass of the subtree.
        offset = data.xpos[body_id] - data.subtree_com[jnp.array(self.__model.body_rootid)[body_id]]
        # vmap over all degrees of freedom of the subtree.
        jacp = jax.vmap(lambda a, b=offset: a[3:] + jnp.cross(a[:3], b))(data.cdof)
        jacp = jax.vmap(jnp.multiply)(jacp, mask)
        jacr = jax.vmap(jnp.multiply)(data.cdof[:, :3], mask)
        jac = jnp.vstack((jacp, jacr))

        return jac

    def get_transform_frame_to_world(self, data: mjx.Data, frame_id: int) -> jnp.array:
        """..."""
        return jnp.concatenate((data.xpos[frame_id], data.xquat[frame_id, [3, 0, 1, 2]]))

    def get_transform(self, data: mjx.Data, source_id: int, dest_id: int) -> jnp.array:
        """..."""
        t1 = self.get_transform_frame_to_world(data, source_id)
        t2 = self.get_transform_frame_to_world(data, dest_id)
        s1, v1 = t1[-1], t1[3:6]
        s2, v2 = t2[-1], t2[3:6]
        delta_q = np.concatenate(
            (
                s1 * s2 + jnp.dot(v1, v2),
                s1 * v2 - s2 * v1 - jnp.cross(v1, v2),
            )
        )
        return jnp.concatenate((t2[:3] - t1[:3], delta_q))

    def integrate(self, data: mjx.Data, velocity: jnp.ndarray, dt: float) -> jnp.array:
        """..."""
        data = data.replace(qvel=velocity)
        return mjx.forward._integrate_pos(self.__model.jnt_type, data.qpos, velocity, dt)

    def integrate_inplace(self, data: mjx.Data, velocity: jnp.ndarray, dt: float) -> mjx.Data:
        """..."""
        return data.replace(qpos=self.integrate(data.qpos, velocity, dt))
