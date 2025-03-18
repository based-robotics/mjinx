import unittest

import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from mjinx.configuration import compute_collision_pairs, update
from mjinx.typing import CollisionPair


class TestCollisionPairs(unittest.TestCase):
    """Test suite for collision pair computation functionality."""

    # XML template for creating test models
    TEST_MODEL_TEMPLATE = """
    <mujoco>
        <worldbody>
            <body name="body1" pos="{pos1}">
                <geom name="geom1" type="{type1}" size="{size1}" />
            </body>
            <body name="body2" pos="{pos2}">
                <joint name="jnt1"/>
                <geom name="geom2" type="{type2}" size="{size2}" />
            </body>
        </worldbody>
    </mujoco>
    """

    def create_test_model(self, type1, type2, pos1, pos2, size1, size2):
        """
        Helper method to create test models with different geometries.

        :param type1: Type of first geometry
        :param type2: Type of second geometry
        :param pos1: Position of first body [x, y, z]
        :param pos2: Position of second body [x, y, z]
        :param size1: Size parameters for first geometry
        :param size2: Size parameters for second geometry
        :return: Tuple of (mjx.Model, mjx.Data)
        """
        xml = self.TEST_MODEL_TEMPLATE.format(
            type1=type1,
            type2=type2,
            pos1=" ".join(map(str, pos1)),
            pos2=" ".join(map(str, pos2)),
            size1=" ".join(map(str, size1 if isinstance(size1, list | tuple) else [size1])),
            size2=" ".join(map(str, size2 if isinstance(size2, list | tuple) else [size2])),
        )
        mjx_model = mjx.put_model(mujoco.MjModel.from_xml_string(xml))
        mjx_data = update(mjx_model, jnp.zeros(1))
        return mjx_model, mjx_data

    def setUp(self):
        """Set up common test models."""
        # Model for sphere-sphere collision tests
        self.sphere_model, self.sphere_data = self.create_test_model(
            "sphere", "sphere", pos1=[0, 0, 0], pos2=[0.8, 0, 0], size1=0.5, size2=0.5
        )

        # Model for box-sphere collision tests
        self.box_sphere_model, self.box_sphere_data = self.create_test_model(
            "box", "sphere", pos1=[0, 0, 0], pos2=[0.8, 0, 0], size1=[0.5, 0.5, 0.5], size2=0.5
        )

    def test_sphere_sphere_collision(self):
        """Test collision detection between two spheres."""
        collision_pairs = [(0, 1)]
        contact = compute_collision_pairs(self.sphere_model, self.sphere_data, collision_pairs)

        # Expected distance: 0.8 - (0.5 + 0.5) = -0.2 (penetration)
        self.assertIsNotNone(contact.dist)
        np.testing.assert_allclose(contact.dist, -0.2, atol=1e-6)
        self.assertIsNotNone(contact.pos)
        np.testing.assert_allclose(contact.pos[0], [0.4, 0, 0], atol=1e-6)

    def test_sphere_sphere_no_collision(self):
        """Test non-colliding spheres."""
        model, data = self.create_test_model("sphere", "sphere", pos1=[0, 0, 0], pos2=[2, 0, 0], size1=0.5, size2=0.5)

        collision_pairs = [(0, 1)]
        contact = compute_collision_pairs(model, data, collision_pairs)

        self.assertIsNotNone(contact.dist)
        np.testing.assert_allclose(contact.dist, 1.0, atol=1e-6)

    def test_box_sphere_collision(self):
        """Test collision detection between a box and a sphere."""
        collision_pairs = [(0, 1)]
        contact = compute_collision_pairs(self.box_sphere_model, self.box_sphere_data, collision_pairs)

        self.assertIsNotNone(contact.dist)
        np.testing.assert_allclose(contact.dist, -0.2, atol=1e-6)

    def test_multiple_collision_pairs(self):
        """Test handling multiple collision pairs simultaneously."""
        # Create model with three bodies
        xml_additional = """
        <mujoco>
            <worldbody>
                <body name="body1" pos="0 0 0">
                    <joint name="jnt1"/>
                    <geom name="geom1" type="sphere" size="0.5" />
                </body>
                <body name="body2" pos="0.8 0 0">
                    <geom name="geom2" type="sphere" size="0.5" />
                </body>
                <body name="body3" pos="0 1.2 0">
                    <geom name="geom3" type="sphere" size="0.5" />
                </body>
            </worldbody>
        </mujoco>
        """
        model = mjx.put_model(mujoco.MjModel.from_xml_string(xml_additional))
        data = update(model, jnp.zeros(1))

        collision_pairs = [(0, 1), (0, 2)]
        contact = compute_collision_pairs(model, data, collision_pairs)

        self.assertIsNotNone(contact.dist)
        self.assertEqual(len(contact.dist), 2)
        np.testing.assert_allclose(contact.dist[0], -0.2, atol=1e-6)  # penetration
        np.testing.assert_allclose(contact.dist[1], 0.2, atol=1e-6)  # separation

    def test_invalid_collision_pairs(self):
        """Test behavior with invalid collision pair indices."""
        with self.assertRaises(IndexError):
            compute_collision_pairs(self.sphere_model, self.sphere_data, [(0, 5)])

    def test_empty_collision_pairs(self):
        """Test behavior with empty collision pairs list."""
        contact = compute_collision_pairs(self.sphere_model, self.sphere_data, [])

        self.assertIsNotNone(contact.dist)
        self.assertEqual(len(contact.dist), 0)
        self.assertIsNotNone(contact.pos)
        self.assertEqual(len(contact.pos), 0)

    def test_contact_frame_orientation(self):
        """Test the orientation of contact frames."""
        model, data = self.create_test_model(
            "sphere",
            "sphere",
            pos1=[0, 0, 0],
            pos2=[0.8, 0.8, 0],
            size1=0.5,
            size2=0.5,  # Diagonal position
        )

        collision_pairs = [(0, 1)]
        contact = compute_collision_pairs(model, data, collision_pairs)

        self.assertIsNotNone(contact.frame)
        expected_direction = np.array([1, 1, 0]) / np.sqrt(2)
        np.testing.assert_allclose(contact.frame[0, 0], expected_direction, atol=1e-6)
