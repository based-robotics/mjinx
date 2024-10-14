import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.barriers._body_position_barrier import JaxPositionBarrier, PositionBarrier
from mjinx.typing import PositionLimitType


class TestPositionBarrier(unittest.TestCase):
    def setUp(self):
        self.model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
        <mujoco>
            <worldbody>
                <body name="body1">
                    <geom name="box1" size=".3"/>
                    <joint name="jnt1" type="free"/>
                </body>
            </worldbody>
        </mujoco>
        """
            )
        )
        self.data = mjx.make_data(self.model)

    def test_initialization(self):
        """Testing component initialization"""
        barrier = PositionBarrier(
            name="test_barrier",
            gain=1.0,
            obj_name="body1",
            p_min=[-1.0, -1.0, -1.0],
            p_max=[1.0, 1.0, 1.0],
        )
        barrier.update_model(self.model)

        self.assertEqual(barrier.name, "test_barrier")
        self.assertEqual(barrier.obj_name, "body1")
        self.assertEqual(barrier.obj_id, 1)
        np.testing.assert_array_equal(barrier.p_min, jnp.array([-1.0, -1.0, -1.0]))
        np.testing.assert_array_equal(barrier.p_max, jnp.array([1.0, 1.0, 1.0]))
        self.assertEqual(barrier.limit_type, PositionLimitType.BOTH)

    def test_update_p_min(self):
        """Testing minimal position updates"""
        barrier = PositionBarrier(
            name="test_barrier",
            gain=1.0,
            obj_name="body1",
            p_min=[-1.0, -1.0, -1.0],
            limit_type="min",
        )
        barrier.update_p_min(0.1)
        np.testing.assert_array_equal(barrier.p_min, 0.1 * jnp.ones(3))

        barrier.update_p_min([-2.0, -2.0, -2.0])
        np.testing.assert_array_equal(barrier.p_min, jnp.array([-2.0, -2.0, -2.0]))

        with self.assertRaises(ValueError):
            barrier.update_p_min([-2.0, -2.0])

        with self.assertWarns(Warning):
            barrier.p_max = [1.0, 1.0, 1.0]

    def test_update_p_max(self):
        """Testing maximal position updates"""
        barrier = PositionBarrier(
            name="test_barrier",
            gain=1.0,
            obj_name="body1",
            p_max=[1.0, 1.0, 1.0],
            limit_type="max",
        )
        barrier.update_p_max(0.1)
        np.testing.assert_array_equal(barrier.p_max, 0.1 * jnp.ones(3))

        barrier.update_p_max([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(barrier.p_max, jnp.array([2.0, 2.0, 2.0]))

        with self.assertRaises(ValueError):
            barrier.update_p_max([2.0, 2.0])

        with self.assertWarns(Warning):
            barrier.p_min = [1.0, 1.0, 1.0]

    def test_limit_type_and_dimension(self):
        """Test limit type detection and dimension of the barrier"""
        barrier_min = PositionBarrier(
            name="test_barrier_min",
            gain=1.0,
            obj_name="body1",
            p_min=[-1.0, -1.0, -1.0],
            limit_type="min",
        )
        self.assertEqual(barrier_min.dim, 3)
        self.assertEqual(barrier_min.limit_type, PositionLimitType.MIN)

        barrier_max = PositionBarrier(
            name="test_barrier_max",
            gain=1.0,
            obj_name="body1",
            p_max=[1.0, 1.0, 1.0],
            limit_type="max",
        )
        self.assertEqual(barrier_max.dim, 3)
        self.assertEqual(barrier_max.limit_type, PositionLimitType.MAX)

        barrier_both = PositionBarrier(
            name="test_barrier_max",
            gain=1.0,
            obj_name="body1",
            p_min=[-1.0, -1.0, -1.0],
            p_max=[1.0, 1.0, 1.0],
            limit_type="both",
        )
        self.assertEqual(barrier_both.dim, 6)
        self.assertEqual(barrier_both.limit_type, PositionLimitType.BOTH)

        with self.assertRaises(ValueError):
            PositionBarrier(
                name="test_barrier_invalid",
                gain=1.0,
                obj_name="body1",
                limit_type="invalid",
            )

    def test_jax_component(self):
        """Test generating jac component"""
        barrier = PositionBarrier(
            name="test_barrier",
            gain=1.0,
            obj_name="body1",
            p_min=[-1.0, -1.0, -1.0],
            p_max=[1.0, 1.0, 1.0],
        )
        barrier.update_model(self.model)

        jax_component = barrier.jax_component

        self.assertIsInstance(jax_component, JaxPositionBarrier)
        self.assertEqual(jax_component.dim, 6)
        np.testing.assert_array_equal(jax_component.vector_gain, jnp.ones(6))
        self.assertEqual(jax_component.body_id, 1)
        np.testing.assert_array_equal(jax_component.p_min, jnp.array([-1.0, -1.0, -1.0]))
        np.testing.assert_array_equal(jax_component.p_max, jnp.array([1.0, 1.0, 1.0]))

    def test_jax_call(self):
        """Test call of the jax barrier function"""
        barrier = JaxPositionBarrier(
            dim=3,
            model=self.model,
            vector_gain=jnp.ones(3),
            gain_fn=lambda x: x,
            mask_idxs=(0, 1, 2),
            safe_displacement_gain=0.0,
            body_id=1,
            p_min=jnp.array([-1.0, -1.0, -1.0]),
            p_max=jnp.array([1.0, 1.0, 1.0]),
        )

        self.data = self.data.replace(qpos=jnp.array([0.0, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0]))
        self.data = mjx.kinematics(self.model, self.data)
        result = barrier(self.data)
        result_compute_barrier = barrier.compute_barrier(self.data)
        np.testing.assert_array_equal(result, result_compute_barrier)
        np.testing.assert_array_almost_equal(result, jnp.array([1.0, 1.5, 0.5, 1.0, 0.5, 1.5]))
