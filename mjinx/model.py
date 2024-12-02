import jax
import jax.numpy as jnp
import mujoco as mj
from jaxlie import SE3, SO3
from mujoco import mjx

from mjinx import typing
from mjinx.components.barriers import Barrier, JointBarrier


class Model:
    def __inin__(
        self,
        v_min: typing.ndarray,
        model: mj.Model | mjx.Model,
        v_max: typing.ndarray,
        fixed_jnts: dict[str, typing.ArrayOrFloat],
        _ignore_closed_kinematics: bool = False,
    ):
        self.model: mjx.Model = mjx.put_model(model) if isinstance(model, mj.Model) else model
        self.data: mjx.Data = mjx.make_data(self.model)

        self.v_min = v_min
        self.v_max = v_max

        # List of tasks and barriers, implied by the model
        self._model_barriers: list[Barrier] = []

        self._fix_jnts(fixed_jnts)
        if not _ignore_closed_kinematics:
            self._parse_closed_loop_kinematics()

    def _fix_jnts(self, fixed_jnts: dict[str, typing.ArrayOrFloat]):
        pass

    def _parse_closed_loop_kinematics(self):
        pass

    def update(self, q: jnp.ndarray) -> mjx.Data:
        """
        Update the MuJoCo data with new joint positions.

        :param model: The MuJoCo model.
        :param q: The new joint positions.
        :return: Updated MuJoCo data.
        """
        data = mjx.make_data(self.model)
        data = data.replace(qpos=q)
        data = mjx.fwd_position(self.model, self.data)
        data = mjx.com_pos(self.model, self.data)

        return data

    def get_frame_jacobian_world_aligned(self, obj_id: int, obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY) -> jnp.ndarray:
        """
        Compute pair of (NV, 3) Jacobians of global point attached to body.

        :param model: The MuJoCo model.
        :param data: The MuJoCo data.
        :param body_id: The ID of the body.
        :return: The Jacobian matrix.
        """

        def fn(carry, b):
            return b if carry is None else b + carry

        body_id: int
        obj_des_pos: jnp.ndarray
        match obj_type:
            case mj.mjtObj.mjOBJ_GEOM:
                body_id = self.model.geom_bodyid[obj_id]
                obj_des_pos = self.data.geom_xpos[obj_id]
            case mj.mjtObj.mjOBJ_SITE:
                body_id = self.model.site_bodyid[obj_id]
                obj_des_pos = self.data.site_xpos[obj_id]
            case _:  # default -- mjOBJ_BODY:
                body_id = obj_id
                obj_des_pos = self.data.xpos[obj_id]

        mask = (jnp.arange(self.model.nbody) == body_id) * 1
        # Puts 1 for all parent links of specified body.
        mask = mjx._src.scan.body_tree(self.model, fn, "b", "b", mask, reverse=True)
        # From all parent links, select only those which add degree of freedoms?..
        mask = mask[jnp.array(self.model.dof_bodyid)] > 0

        # Subtree_com is the center of mass of the subtree.
        offset = obj_des_pos - self.data.subtree_com[jnp.array(self.model.body_rootid)[body_id]]
        # vmap over all degrees of freedom of the subtree.
        jacp = jax.vmap(lambda a, b=offset: a[3:] + jnp.cross(a[:3], b))(self.data.cdof)
        jacp = jax.vmap(jnp.multiply)(jacp, mask)
        jacr = jax.vmap(jnp.multiply)(self.data.cdof[:, :3], mask)

        return jnp.vstack((jacp.T, jacr.T)).T

    def get_frame_jacobian_local(self, obj_id: int, obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY) -> jax.Array:
        """
        Compute pair of (NV, 3) Jacobians of global point attached to body in local frame.

        :param model: The MuJoCo model.
        :param data: The MuJoCo data.
        :param body_id: The ID of the body.
        :return: The Jacobian matrix in local frame.
        """

        def fn(carry, b):
            return b if carry is None else b + carry

        body_id: int
        obj_des_pos: jnp.ndarray
        obj_des_rot: jnp.ndarray
        match obj_type:
            case mj.mjtObj.mjOBJ_GEOM:
                body_id = self.model.geom_bodyid[obj_id]
                obj_des_pos = self.data.geom_xpos[obj_id]
                obj_des_rot = self.data.geom_xmat[obj_id]
            case mj.mjtObj.mjOBJ_SITE:
                body_id = self.model.site_bodyid[obj_id]
                obj_des_pos = self.data.site_xpos[obj_id]
                obj_des_rot = self.data.site_xmat[obj_id]
            case _:  # default -- mjOBJ_BODY:
                body_id = obj_id
                obj_des_pos = self.data.xpos[obj_id]
                obj_des_rot = self.data.xmat[obj_id]

        mask = (jnp.arange(self.model.nbody) == body_id) * 1
        # Puts 1 for all parent links of specified body.
        mask = mjx._src.scan.body_tree(self.model, fn, "b", "b", mask, reverse=True)
        # From all parent links, select only those which add degree of freedoms?..
        mask = mask[jnp.array(self.model.dof_bodyid)] > 0

        # Subtree_com is the center of mass of the subtree.
        offset = obj_des_pos - self.data.subtree_com[jnp.array(self.model.body_rootid)[body_id]]

        # Get rotation matrix, which describes rotation of local frame
        R_inv = obj_des_rot.reshape(3, 3).T

        # vmap over all degrees of freedom of the subtree.
        jacp = jax.vmap(lambda a, b=offset, R=R_inv: R @ (a[3:] + jnp.cross(a[:3], b)))(self.data.cdof)
        jacp = jax.vmap(jnp.multiply)(jacp, mask)
        jacr = jax.vmap(jnp.multiply)(self.data.cdof[:, :3] @ R_inv.T, mask)

        return jnp.vstack((jacp.T, jacr.T)).T

    def get_transform_frame_to_world(self, frame_id: int) -> SE3:
        """
        Get the transformation from frame to world coordinates.

        :param model: The MuJoCo model.
        :param data: The MuJoCo data.
        :param frame_id: The ID of the frame.
        :return: The SE3 transformation.
        """
        return SE3.from_rotation_and_translation(
            SO3.from_quaternion_xyzw(self.data.xquat[frame_id, [1, 2, 3, 0]]),
            self.data.xpos[frame_id],
        )

    def get_transform(self, source_id: int, dest_id: int) -> SE3:
        """
        Get the transformation between two frames.

        :param model: The MuJoCo model.
        :param data: The MuJoCo data.
        :param source_id: The ID of the source frame.
        :param dest_id: The ID of the destination frame.
        :return: The SE3 transformation from source to destination.
        """
        return self.get_transform_frame_to_world(dest_id) @ self.get_transform_frame_to_world(source_id)

    def integrate(self, q0: jnp.ndarray, velocity: jnp.ndarray, dt: jnp.ndarray | float) -> jnp.ndarray:
        """
        Integrate the joint positions given initial position, velocity, and time step.

        :param model: The MuJoCo model.
        :param q0: The initial joint positions.
        :param velocity: The joint velocities.
        :param dt: The time step.
        :return: The integrated joint positions.
        """
        return mjx._src.forward._integrate_pos(self.model.jnt_type, q0, velocity, dt)

    def get_configuration_limit(self, limit: jnp.ndarray | float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the configuration limits for the model.

        :param model: The MuJoCo model.
        :param limit: The limit value(s).
        :return: A tuple of arrays representing the lower and upper bounds.
        """
        # -limit <= v <- limit
        limit_array = jnp.ones(self.model.nv) * limit if isinstance(limit, float) else limit

        return (
            jnp.vstack((-1 * jnp.eye(self.model.nv), jnp.eye(self.model.nv))),
            jnp.concatenate((limit_array, limit_array)),
        )

    def geom_point_jacobian(self, point: jnp.ndarray, body_id: jnp.ndarray) -> jnp.ndarray:
        jacp, jacr = mjx._src.support.jac(self.model, self.data, point, body_id)
        return jnp.vstack((jacp.T, jacr.T)).T
