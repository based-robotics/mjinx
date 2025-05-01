from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import final

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
from jaxlie import SE3, SO3
from mujoco.mjx._src import math, mesh
from mujoco.mjx._src.collision_convex import sphere_convex
from mujoco.mjx._src.collision_sdf import _cylinder, _ellipsoid, _optim, _sphere
from mujoco.mjx._src.collision_types import GeomInfo

from mjinx.components.barriers._obj_barrier import JaxObjBarrier, ObjBarrier
from mjinx.typing import ArrayOrFloat, ndarray


def sphere_sphere(sphere1: GeomInfo, sphere2: GeomInfo) -> jnp.ndarray:
    """Calculates the distance between two spheres."""
    return jnp.linalg.norm(sphere1.pos - sphere2.pos) - (sphere1.size[0] + sphere2.size[0])


def sphere_capsule(sphere: GeomInfo, capsule: GeomInfo) -> jnp.ndarray:
    """Calculates distance between a sphere and a capsule."""
    axis, length = capsule.mat[:, 2], capsule.size[1]
    segment = axis * length
    pt = math.closest_segment_point(capsule.pos - segment, capsule.pos + segment, sphere.pos)
    return jnp.linalg.norm(sphere.pos - pt) - (sphere.size[0] + capsule.size[0])


def sphere_cylinder(sphere: GeomInfo, cylinder: GeomInfo) -> jnp.ndarray:
    """Calculates distance between a sphere and a cylinder.

    Cylinder is aligned so its local z-axis is the length direction,
    oriented in world coords by R_c.  Let v = R_c^T (p_s - p_c) be the
    sphere-to-cylinder vector in the cylinder's frame, with
      v_xy = v[:2],  z = v[2].

    Then the point-cylinder distance is
      d_pt = sqrt(max(||v_xy|| - r, 0)^2 + max(|z| - h, 0)^2),
    and the signed sphere-cylinder distance is
      d = d_pt - R.
    """
    # vector from cylinder center to sphere center, expressed in cylinder frame
    v_world = jnp.array(sphere.pos) - jnp.array(cylinder.pos)
    v_local = jnp.matmul(cylinder.mat.T, v_world)

    # radial distance in the xy-plane of the cylinder
    rho = jnp.linalg.norm(v_local[:2], axis=-1)

    # amount outside cylinder radius, and outside end caps
    dr = jnp.maximum(rho - cylinder.size[0], 0.0)
    dz = jnp.maximum(jnp.abs(v_local[2]) - cylinder.size[1], 0.0)

    # distance from point to cylinder surface
    d_point_cyl = jnp.sqrt(dr**2 + dz**2)

    # subtract sphere radius
    return d_point_cyl - sphere.size[0]


def sphere_ellipsoid(sphere: GeomInfo, ellipsoid: GeomInfo) -> jnp.ndarray:
    """Calculates distance between a sphere and an ellipsoid.

    Sphere: center p_s, radius R_s.
    Ellipsoid: center p_e, principal radii (X, Y, Z), oriented by R_e.
    Positive => separated; negative => overlapping.

    We find t>=0 solving
        φ(t) = sum_i [v_i^2 * a_i^2 / (a_i^2 + t)^2] - 1 = 0
    via Newton, where v = R_e^T (p_s - p_e) in ellipsoid frame and
    a_i are the radii.
    Closest point q has components q_i = a_i^2 * v_i / (a_i^2 + t).
    Distance = ‖v - q‖₂ - R_s.
    """
    # stack radii and their squares
    a2 = ellipsoid.size**2

    # vector from ellipsoid center to sphere center, in world frame
    v_world = jnp.array(sphere.pos) - jnp.array(ellipsoid.pos)

    # express it in the ellipsoid's local frame
    v_local = jnp.einsum("...ij,...j->...i", ellipsoid.mat.transpose(-1, -2), v_world)

    # initial guess for t (ensure non-negative)
    # use max(||v|| - min(a), 0)
    norm_v = jnp.linalg.norm(v_local, axis=-1)
    t0 = jnp.maximum(norm_v - jnp.min(ellipsoid.size), 0.0)

    def newton_step(t):
        # φ(t) and φ'(t)
        # expand dims so broadcasting works over batch dims
        t_exp = jnp.expand_dims(t, axis=-1)
        denom = (a2 + t_exp) ** 2  # shape (...,3)
        φ = jnp.sum(v_local**2 * a2 / denom, axis=-1) - 1.0
        φp = -2.0 * jnp.sum(v_local**2 * a2 / ((a2 + t_exp) ** 3), axis=-1)
        t_new = t - φ / φp
        return jnp.maximum(t_new, 0.0)

    # run fixed number of Newton iterations
    t = jax.lax.fori_loop(0, 10, lambda i, tv: newton_step(tv), t0)

    # compute closest point on ellipsoid surface, in local frame
    t_exp = jnp.expand_dims(t, axis=-1)
    q_local = v_local * (a2 / (a2 + t_exp))

    # distance from sphere center to ellipsoid surface
    d_pt = jnp.linalg.norm(v_local - q_local, axis=-1)

    # signed distance sphere–ellipsoid
    return d_pt - sphere.size[0]


def sphere_box(sphere: GeomInfo, box: GeomInfo) -> jnp.ndarray:
    """Calculates distance between a sphere and a box.

    Let B be the box centered at p_b, with half-extents (X/2, Y/2, Z/2)
    in its own local frame, oriented by rotation matrix R_b.
    Then the signed distance d is
        d = ‖ max(|R_b.T @ (p_s - p_b)| - half_extents, 0) ‖_2  - R

    If d > 0, sphere is separated; if d < 0, sphere intersects or contains part of the box.
    """
    # half-extents of box
    half_extents = box.size

    # vector from box center to sphere center, in world coords
    v_world = jnp.array(sphere.pos) - jnp.array(box.pos)

    # express that vector in box's local frame
    v_local = jnp.matmul(box.mat.T, v_world)

    # compute per-axis distance outside the box (zeros inside)
    outside_dist = jnp.maximum(jnp.abs(v_local) - half_extents, 0.0)

    # Euclidean distance from point to box surface
    d_point_box = jnp.linalg.norm(outside_dist)

    # subtract sphere radius to get sphere-to-box distance
    return d_point_box - sphere.size[0]


_COLLISION_FUNC = {
    mj.mjtGeom.mjGEOM_SPHERE: sphere_sphere,
    mj.mjtGeom.mjGEOM_CAPSULE: sphere_capsule,
    mj.mjtGeom.mjGEOM_CYLINDER: sphere_cylinder,
    mj.mjtGeom.mjGEOM_ELLIPSOID: sphere_ellipsoid,
    mj.mjtGeom.mjGEOM_BOX: sphere_box,
}


@jdc.pytree_dataclass
class JaxGeomBarrier(JaxObjBarrier):
    geom_type: jdc.Static[mj.mjtGeom]
    geom_frame: SE3
    geom_size: jnp.ndarray
    d_min: float

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        xpos = self.geom_frame.translation()
        xmat = self.geom_frame.rotation().as_matrix()

        collision_pair = self.geom_type

        geom_info = GeomInfo(xpos, xmat, self.geom_size)
        sphere_info = GeomInfo(
            self.get_pos(data),
            self.get_rotation(data).as_matrix(),
            jnp.array([self.d_min, 0, 0]),
        )

        collision_fn = _COLLISION_FUNC[collision_pair]

        return collision_fn(sphere_info, geom_info).reshape(1)


class GeomBarrier(ObjBarrier[JaxGeomBarrier]):
    JaxComponentType: type = JaxGeomBarrier
    d_min: float
    _geom_frame: SE3
    _geom_size: jnp.ndarray
    _geom_type: jdc.Static[mj.mjtGeom]

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        obj_name: str,
        obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        geom_type: mj.mjtGeom = mj.mjtGeom.mjGEOM_SPHERE,
        geom_frame: SE3 | Sequence | ndarray | None = None,
        geom_size: jnp.ndarray | Sequence | float | None = None,
        d_min: float = 0.0,
    ):
        """
        Initialize the PositionBarrier object.

        :param name: The name of the barrier.
        :param gain: The gain for the barrier function.
        :param obj_name: The name of the object (body, geometry, or site) to which this barrier applies.
        :param obj_type: The type of the object, supported all geometries taht collide with sphere, except mjGEOM_CONVEX.
        :param gain_fn: A function to compute the gain dynamically.
        :param safe_displacement_gain: The gain for computing safe displacements.
        """
        super().__init__(name, gain, obj_name, obj_type, gain_fn, safe_displacement_gain, jnp.ones(1))
        self._dim = 1

        self._geom_type = geom_type
        self.d_min = d_min
        if geom_frame is not None:
            self.update_geom_frame(geom_frame)
        if geom_size is not None:
            self.geom_size = geom_size

    @property
    def geom_type(self) -> mj.mjtGeom:
        """
        Get the type of the geometry.

        :return: The type of the geometry.
        """
        return self._geom_type

    @property
    def geom_frame(self) -> SE3:
        """
        Get the state of the geometry frame.

        :return: The current target frame as an SE3 object.
        """
        return self._geom_frame

    @geom_frame.setter
    def geom_frame(self, value: SE3 | Sequence | ndarray):
        """
        Set the target frame for the task.

        :param value: The new target frame, either as an SE3 object or a sequence of values.
        """
        self.update_geom_frame(value)

    def update_geom_frame(self, target_frame: SE3 | Sequence | ndarray):
        """
        Update the target frame for the task.

        This method allows setting the target frame using either an SE3 object
        or a sequence of values representing the frame.

        :param target_frame: The new target frame, either as an SE3 object or a sequence of values.
        :raises ValueError: If the provided sequence doesn't have the correct length.
        """
        if not isinstance(target_frame, SE3):
            target_frame_jnp = jnp.array(target_frame)
            if target_frame_jnp.shape[-1] != SE3.parameters_dim:
                raise ValueError(
                    "Target frame provided via array must have length 7 (xyz + quaternion with scalar first)"
                )

            xyz, quat = target_frame_jnp[..., :3], target_frame_jnp[..., 3:]
            geom_frame = SE3.from_rotation_and_translation(
                SO3.from_quaternion_xyzw(
                    quat[..., [1, 2, 3, 0]],
                ),
                xyz,
            )
        else:
            geom_frame = target_frame
        self._geom_frame = geom_frame

    @property
    def geom_size(self) -> jnp.ndarray:
        """
        Get the size of the geometry.

        :return: The size of the geometry.
        """
        return self._geom_size

    @geom_size.setter
    def geom_size(self, value: jnp.ndarray | Sequence):
        """
        Set the size of the geometry.

        :param value: The new size of the geometry.
        """
        geom_size = jnp.array(value)
        self._geom_size = jnp.zeros(3).at[: len(geom_size)].set(geom_size)
