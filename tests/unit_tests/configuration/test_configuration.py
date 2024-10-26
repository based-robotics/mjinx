import unittest

import jax
import jax.numpy as jnp
import mujoco as mj
from jaxlie import SE3
from mujoco import mjx

from mjinx.configuration import (
    attitude_jacobian,
    get_configuration_limit,
    get_distance,
    get_frame_jacobian_local,
    get_frame_jacobian_world_aligned,
    get_joint_zero,
    get_transform,
    get_transform_frame_to_world,
    integrate,
    jac_dq2v,
    joint_difference,
    skew_symmetric,
    update,
)

# Import the functions to be tested
from mjinx.typing import CollisionPair


class TestConfiguration(unittest.TestCase):
    def setUp(self):
        # Create a simple MuJoCo model for testing
        self.model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
        <mujoco>
            <worldbody>
                <body name="body1">
                    <geom name="box1" size=".3"/>
                    <site name="site1" pos="0.1 0.1 0.1"/>
                    <joint name="jnt1" type="hinge" axis="1 -1 0" range="-1 2"/>
                    <body name="body2">
                        <joint name="jnt2" type="ball" range="0 2"/>
                        <geom name="box2" pos=".6 .6 .6" size=".3"/>
                        <body name="body3">
                            <joint name="jnt3" type="hinge" axis="1 -1 0" pos=".6 .6 .6" range="-1 2"/>
                            <geom name="box3" pos="1.2 1.2 1.2" size=".3"/>
                        </body>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """
            )
        )
        self.nq = self.model.nq
        self.q = jnp.array([0, 0, 0, 0, 1, 0], dtype=jnp.float32)  # Example joint position

    def test_update(self):
        data = update(self.model, self.q)
        self.assertIsInstance(data, mjx.Data)
        self.assertTrue(jnp.allclose(data.qpos, self.q))

    def test_get_frame_jacobian_world_aligned(self):
        self.data = update(self.model, self.q)
        # Test for body
        jacobian_body = get_frame_jacobian_world_aligned(self.model, self.data, 1, mj.mjtObj.mjOBJ_BODY)
        self.assertEqual(jacobian_body.shape, (self.model.nv, 6))

        # Test for geom
        jacobian_geom = get_frame_jacobian_world_aligned(self.model, self.data, 0, mj.mjtObj.mjOBJ_GEOM)
        self.assertEqual(jacobian_geom.shape, (self.model.nv, 6))

        # Test for site
        jacobian_site = get_frame_jacobian_world_aligned(self.model, self.data, 0, mj.mjtObj.mjOBJ_SITE)
        self.assertEqual(jacobian_site.shape, (self.model.nv, 6))

        # Check that the jacobian is zero for parts of the kinematic chain that don't affect the object
        # For example, the jacobian for body1 should be zero w.r.t. jnt3
        jacobian_body1 = get_frame_jacobian_world_aligned(self.model, self.data, 0, mj.mjtObj.mjOBJ_BODY)
        self.assertTrue(jnp.allclose(jacobian_body1[:, -1], 0))  # Last column should be zero (corresponding to jnt3)

    def test_get_frame_jacobian_local(self):
        # TODO: write tests with particular numbers
        self.data = update(self.model, self.q)
        # Test for body
        jacobian_body = get_frame_jacobian_local(self.model, self.data, 1, mj.mjtObj.mjOBJ_BODY)
        self.assertEqual(jacobian_body.shape, (self.model.nv, 6))

        # Test for geom
        jacobian_geom = get_frame_jacobian_local(self.model, self.data, 0, mj.mjtObj.mjOBJ_GEOM)
        self.assertEqual(jacobian_geom.shape, (self.model.nv, 6))

        # Test for site
        jacobian_site = get_frame_jacobian_local(self.model, self.data, 0, mj.mjtObj.mjOBJ_SITE)
        self.assertEqual(jacobian_site.shape, (self.model.nv, 6))

        # Check that the jacobian is zero for parts of the kinematic chain that don't affect the object
        # For example, the jacobian for body1 should be zero w.r.t. jnt3
        jacobian_body1 = get_frame_jacobian_local(self.model, self.data, 0, mj.mjtObj.mjOBJ_BODY)
        self.assertTrue(jnp.allclose(jacobian_body1[:, -1], 0))  # Last column should be zero (corresponding to jnt3)

        # Check that the local frame jacobian is different from the world-aligned jacobian
        jacobian_world = get_frame_jacobian_world_aligned(self.model, self.data, 2, mj.mjtObj.mjOBJ_BODY)
        self.assertFalse(jnp.allclose(jacobian_body, jacobian_world))

    def test_get_transform_frame_to_world(self):
        data = update(self.model, self.q)
        transform = get_transform_frame_to_world(self.model, data, 1)
        self.assertIsInstance(transform, SE3)

    def test_get_transform(self):
        data = update(self.model, self.q)
        transform = get_transform(self.model, data, 0, 1)
        self.assertIsInstance(transform, SE3)

    def test_integrate(self):
        velocity = jnp.array([0.1])
        dt = jnp.array(0.1)
        q_new = integrate(self.model, self.q, velocity, dt)
        self.assertEqual(q_new.shape, self.q.shape)

    def test_get_configuration_limit(self):
        limit = 1.0
        A, b = get_configuration_limit(self.model, limit)
        self.assertEqual(A.shape, (2 * self.model.nv, self.model.nv))
        self.assertEqual(b.shape, (2 * self.model.nv,))

    def test_get_joint_zero(self):
        zero_config = get_joint_zero(self.model)
        self.assertEqual(zero_config.shape, (self.model.nq,))

    def test_joint_difference(self):
        q1 = self.q.copy()
        q2 = jnp.array([0.1, 1.0, 0.0, 0.0, 0.0, -0.2])
        diff = joint_difference(self.model, q1, q2)
        self.assertEqual(diff.shape, (self.model.nv,))

    # Additional tests for edge cases and different joint types
    def test_get_joint_zero_complex_model(self):
        complex_model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
            <mujoco>
            <worldbody>
                <body name="body1" pos="0 0 0">
                <geom name="body1" size=".3"/>
                    <joint name="free_joint" type="free"/>
                    <body name="body2">
                        <geom name="body2" size=".3"/>
                        <joint name="ball_joint" type="ball"/>
                        <body name="body3">
                            <geom name="body3" size=".3"/>
                            <joint name="hinge_joint" type="hinge"/>
                            <body name="body4">
                                <geom name="body4" size=".3"/>
                                <joint name="slide_joint" type="slide"/>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
            </mujoco>
        """
            )
        )
        zero_config = get_joint_zero(complex_model)
        expected_length = 7 + 4 + 1 + 1  # free + ball + hinge + slide
        self.assertEqual(zero_config.shape, (expected_length,))

    def test_joint_difference_complex_model(self):
        complex_model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
            <mujoco>
            <worldbody>
                <body name="body1" pos="0 0 0">
                <geom name="body1" size=".3"/>
                    <joint name="free_joint" type="free"/>
                    <body name="body2">
                        <geom name="body2" size=".3"/>
                        <joint name="ball_joint" type="ball"/>
                        <body name="body3">
                            <geom name="body3" size=".3"/>
                            <joint name="hinge_joint" type="hinge"/>
                            <body name="body4">
                                <geom name="body4" size=".3"/>
                                <joint name="slide_joint" type="slide"/>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
            </mujoco>
        """
            )
        )
        q1 = jnp.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.5, 0.2])
        q2 = jnp.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0.7, 0.3])
        diff = joint_difference(complex_model, q1, q2)
        expected_length = 6 + 3 + 1 + 1  # free + ball + hinge + slide
        self.assertEqual(diff.shape, (expected_length,))

    def test_get_configuration_limit_array(self):
        limit_array = jnp.array([0.5, 1.0, 1.5])
        model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
            <mujoco>
            <worldbody>
                <body name="body1">
                    <geom name="body1" size=".3"/>
                    <joint name="joint1" type="slide"/>
                </body>
                <body name="body2">
                    <geom name="body2" size=".3"/>
                    <joint name="joint2" type="hinge"/>
                </body>
                <body name="body3">
                    <geom name="body3" size=".3"/>
                    <joint name="joint3" type="ball"/>
                </body>
            </worldbody>
            </mujoco>
            """
            )
        )
        A, b = get_configuration_limit(model, limit_array)
        self.assertEqual(A.shape, (10, 5))
        self.assertTrue(jnp.allclose(b, jnp.array([0.5, 1.0, 1.5, 0.5, 1.0, 1.5])))

    def test_get_distance(self) -> None:
        self.model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
       <mujoco>
           <worldbody>
               <geom name="floor" pos="0 0 0" size="5 5 0.1" type="plane"/>
               <body name="body1" pos="0 0 1">
                   <joint type="free"/>
                   <geom name="sphere1" pos="0 0 0" size="0.1" type="sphere"/>
               </body>
               <body name="body2" pos="1 0 1">
                   <joint type="free"/>
                   <geom name="box1" pos="0 0 0" size="0.1 0.1 0.1" type="box"/>
               </body>
               <body name="body3" pos="0 1 1">
                   <joint type="free"/>
                   <geom name="capsule1" pos="0 0 0" size="0.1 0.2" type="capsule"/>
               </body>
           </worldbody>
       </mujoco>
       """
            )
        )
        self.data = mjx.make_data(self.model)
        self.data = mjx.kinematics(self.model, self.data)

        # Define collision pairs
        collision_pairs: list[CollisionPair] = [
            (1, 2),  # sphere and box
            (1, 3),  # sphere and capsule
            (2, 3),  # box and capsule
            (0, 1),  # floor and sphere
            (0, 2),  # floor and box
            (0, 3),  # floor and capsule
        ]

        # Get distances
        distances = get_distance(self.model, self.data, collision_pairs)

        # Check if the output is a jax array
        self.assertIsInstance(distances, jax.Array)

        # Check if the number of distances matches the number of collision pairs
        self.assertEqual(len(distances), len(collision_pairs))

        # Check if all distances are non-negative
        self.assertTrue(jnp.all(distances >= 0))

        # Check specific distances
        # FIXME: collisions with boxes is not functioning properly
        # self.assertAlmostEqual(distances[0], 0.8, delta=0.1)  # sphere to box
        # self.assertAlmostEqual(distances[2], jnp.sqrt(2) - 0.2, delta=0.1)  # box to capsule

        self.assertAlmostEqual(distances[1], 0.8, delta=1e-3)  # sphere to capsule
        self.assertAlmostEqual(distances[3], 0.9, delta=1e-3)  # floor to sphere
        self.assertAlmostEqual(distances[4], 0.9, delta=1e-3)  # floor to box
        self.assertAlmostEqual(distances[5], 0.7, delta=1e-3)  # floor to capsule

    def test_get_distance_with_hfield(self) -> None:
        hfield_model = mj.MjModel.from_xml_string(
            """
       <mujoco>
           <asset>
               <hfield name="hf" nrow="10" ncol="10" size="1 1 0.1 0.1"/>
           </asset>
           <worldbody>
               <geom name="hfield" pos="0 0 0" size="1 1 0.1" type="hfield" hfield="hf"/>
               <body name="body1" pos="0 0 1">
                   <joint type="free"/>
                   <geom name="sphere1" pos="0 0 0" size="0.1" type="sphere"/>
               </body>
           </worldbody>
       </mujoco>
       """
        )
        hfield_data = mjx.make_data(hfield_model)
        hfield_data = mjx.kinematics(hfield_model, hfield_data)

        collision_pairs: list[CollisionPair] = [(0, 1)]  # hfield and sphere

        with self.assertRaises(NotImplementedError):
            get_distance(hfield_model, hfield_data, collision_pairs)

    def test_skew_symmetric(self):
        v = jnp.array([1.0, 2.0, 3.0])
        result = skew_symmetric(v)
        expected = jnp.array(
            [
                [0.0, -3.0, 2.0],
                [3.0, 0.0, -1.0],
                [-2.0, 1.0, 0.0],
            ]
        )
        self.assertTrue(jnp.allclose(result, expected))

    def test_attitude_jacobian(self):
        q = jnp.array([0.5, 0.5, 0.5, 0.5])  # Example quaternion
        result = attitude_jacobian(q)
        expected = jnp.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
            ]
        )
        self.assertTrue(jnp.allclose(result, expected))

    def test_jac_dq2v(self):
        # Create a model with different joint types
        model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
                <mujoco>
                    <worldbody>
                        <body name="body1">
                            <geom name="geom1" size="0.1"/>
                            <joint name="free_joint" type="free"/>
                            <body name="body2">
                                <geom name="geom2" size="0.1"/>
                                <joint name="ball_joint" type="ball"/>
                                <body name="body3">
                                    <geom name="geom3" size="0.1"/>
                                    <joint name="hinge_joint" type="hinge"/>
                                </body>
                            </body>
                        </body>
                    </worldbody>
                </mujoco>
                """
            )
        )

        # Example configuration (adjust based on your model's degrees of freedom)
        q = jnp.array(
            [
                1.0,
                2.0,
                3.0,
                1.0,
                0.0,
                0.0,
                0.0,  # Free joint (3 pos + 4 quat)
                1.0,
                0.0,
                0.0,
                0.0,  # Ball joint (4 quat)
                0.5,
            ]
        )  # Hinge joint (1 angle)

        result = jac_dq2v(model, q)

        # Check the shape of the result
        self.assertEqual(result.shape, (model.nq, model.nv))

        # Check specific elements (you may need to adjust these based on your implementation)
        # Free joint
        self.assertTrue(jnp.allclose(result[:3, :3], jnp.eye(3)))  # Position part
        self.assertTrue(jnp.allclose(result[3:7, 3:6], attitude_jacobian(q[3:7])))  # Rotation part

        # Ball joint
        self.assertTrue(jnp.allclose(result[7:11, 6:9], attitude_jacobian(q[7:11])))

        # Hinge joint
        self.assertEqual(result[11, 9], 1.0)

        # Check that all other elements are zero
        mask = jnp.ones_like(result, dtype=bool)
        mask = mask.at[:3, :3].set(False)
        mask = mask.at[3:7, 3:6].set(False)
        mask = mask.at[7:11, 6:9].set(False)
        mask = mask.at[11, 9].set(False)
        self.assertTrue(jnp.allclose(result[mask], 0.0))
