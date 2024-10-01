import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.barriers._self_collision_barrier import JaxSelfCollisionBarrier, SelfCollisionBarrier
from mjinx.configuration import get_distance, sorted_pair


class TestSelfCollisionBarrier(unittest.TestCase):
    def setUp(self):
        self.model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
        <mujoco>
            <worldbody>
                <body name="body1">
                    <joint name="j1"/>
                    <geom name="geom1" size="0.1" type="sphere"/>
                </body>
                <body name="body2">
                    <joint name="j2"/>
                    <geom name="geom2" size="0.1" type="sphere"/>
                    <body name="body3">
                        <joint name="j3"/>
                        <geom name="geom3" size="0.1" type="sphere"/>
                        <body name="body4">
                            <joint name="j4"/>
                            <geom name="geom4" size="0.1" type="sphere" contype="2" conaffinity="2"/>
                        </body>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """
            )
        )
        self.data = mjx.make_data(self.model)

    def test_initialization(self):
        barrier = SelfCollisionBarrier(
            name="test_barrier",
            gain=1.0,
            d_min=0.1,
            collision_bodies=["body1", "body2"],
            excluded_collisions=[("body1", "body2")],
        )
        barrier.update_model(self.model)

        self.assertEqual(barrier.name, "test_barrier")
        self.assertEqual(barrier.d_min, 0.1)
        self.assertEqual(barrier.collision_bodies, ["body1", "body2"])
        self.assertEqual(barrier.exclude_collisions, {sorted_pair(1, 2)})

    def test_generate_collision_pairs(self):
        barrier = SelfCollisionBarrier(
            name="test_barrier",
            gain=1.0,
            d_min=0.1,
        )
        barrier.update_model(self.model)

        expected_pairs = [(0, 1), (0, 2)]
        self.assertEqual(barrier.collision_pairs, expected_pairs)

    def test_generate_collision_pairs_with_exclusion(self):
        barrier = SelfCollisionBarrier(
            name="test_barrier",
            gain=1.0,
            d_min=0.1,
            excluded_collisions=[("body1", "body2")],
        )
        barrier.update_model(self.model)

        expected_pairs = [(0, 2)]
        self.assertEqual(barrier.collision_pairs, expected_pairs)

    def test_body2id(self):
        barrier = SelfCollisionBarrier(
            name="test_barrier",
            gain=1.0,
            d_min=0.1,
        )
        barrier.update_model(self.model)

        self.assertEqual(barrier._SelfCollisionBarrier__body2id("body1"), 1)
        self.assertEqual(barrier._SelfCollisionBarrier__body2id(1), 1)

        with self.assertRaises(ValueError):
            barrier._SelfCollisionBarrier__body2id(1.5)

    def test_validate_body_pair(self):
        barrier = SelfCollisionBarrier(
            name="test_barrier",
            gain=1.0,
            d_min=0.1,
        )
        barrier.update_model(self.model)

        self.assertTrue(barrier._SelfCollisionBarrier__validate_body_pair(1, 2))
        self.assertTrue(barrier._SelfCollisionBarrier__validate_body_pair(1, 3))
        self.assertTrue(barrier._SelfCollisionBarrier__validate_body_pair(2, 4))

        self.assertFalse(barrier._SelfCollisionBarrier__validate_body_pair(2, 3))
        self.assertFalse(barrier._SelfCollisionBarrier__validate_body_pair(3, 4))

    def test_validate_geom_pair(self):
        barrier = SelfCollisionBarrier(
            name="test_barrier",
            gain=1.0,
            d_min=0.1,
        )
        barrier.update_model(self.model)

        self.assertTrue(barrier._SelfCollisionBarrier__validate_geom_pair(0, 1))
        self.assertTrue(barrier._SelfCollisionBarrier__validate_geom_pair(0, 2))
        self.assertTrue(barrier._SelfCollisionBarrier__validate_geom_pair(1, 2))

        for i in range(self.model.ngeom - 1):
            self.assertFalse(barrier._SelfCollisionBarrier__validate_geom_pair(3, i))

    def test_d_min_vec(self):
        barrier = SelfCollisionBarrier(
            name="test_barrier",
            gain=1.0,
            d_min=0.1,
        )
        barrier.update_model(self.model)

        np.testing.assert_array_equal(barrier.d_min_vec, jnp.ones(barrier.dim) * 0.1)

    def test_jax_component(self):
        barrier = SelfCollisionBarrier(
            name="test_barrier",
            gain=1.0,
            d_min=0.1,
        )
        barrier.update_model(self.model)

        jax_component = barrier.jax_component

        self.assertIsInstance(jax_component, JaxSelfCollisionBarrier)
        self.assertEqual(jax_component.dim, 2)
        np.testing.assert_array_equal(jax_component.vector_gain, jnp.ones(jax_component.dim))
        np.testing.assert_array_equal(jax_component.d_min_vec, jnp.ones(jax_component.dim) * 0.1)
        self.assertEqual(len(jax_component.collision_pairs), jax_component.dim)

    def test_call(self):
        collision_pairs = [(0, 1)]
        barrier = JaxSelfCollisionBarrier(
            dim=1,
            model=self.model,
            vector_gain=jnp.ones(1),
            gain_fn=lambda x: x,
            mask_idxs=(0,),
            safe_displacement_gain=0.0,
            d_min_vec=jnp.array([0.1]),
            collision_pairs=collision_pairs,
        )

        result = barrier(self.data)
        expected = get_distance(self.model, self.data, collision_pairs) - barrier.d_min_vec
        np.testing.assert_array_almost_equal(result, expected)
