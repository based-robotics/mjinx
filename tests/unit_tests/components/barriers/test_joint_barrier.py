import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.barriers import JaxJointBarrier, JointBarrier
from mjinx.configuration import joint_difference


class TestJointBarrier(unittest.TestCase):
    def setUp(self):
        self.model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
        <mujoco>
            <worldbody>
                <body name="body1">
                    <joint name="joint1" type="hinge" range="-0.1 0.1"/>
                    <geom name="geom1" size="0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
            )
        )
        self.data = mjx.make_data(self.model)

    def test_initialization(self):
        barrier = JointBarrier(
            name="test_barrier",
            gain=1.0,
            q_min=[-1.0],
            q_max=[1.0],
        )
        barrier.update_model(self.model)

        np.testing.assert_array_equal(barrier.q_min, jnp.array([-1.0]))
        np.testing.assert_array_equal(barrier.q_max, jnp.array([1.0]))
        self.assertEqual(barrier.dim, 2 * self.model.nv)

    def test_update_q_min(self):
        barrier = JointBarrier(
            name="test_barrier",
            gain=1.0,
            q_min=[-1.0],
            q_max=[1.0],
        )
        barrier.update_model(self.model)

        barrier.update_q_min(jnp.array([-2.0]))
        np.testing.assert_array_equal(barrier.q_min, jnp.array([-2.0]))

        with self.assertRaises(ValueError):
            barrier.update_q_min(jnp.array([-2.0, -2.0]))

    def test_update_q_max(self):
        barrier = JointBarrier(
            name="test_barrier",
            gain=1.0,
            q_min=[-1.0],
            q_max=[1.0],
        )
        barrier.update_model(self.model)

        barrier.update_q_max(jnp.array([2.0]))
        np.testing.assert_array_equal(barrier.q_max, jnp.array([2.0]))

        with self.assertRaises(ValueError):
            barrier.update_q_max(jnp.array([2.0, 2.0]))

    def test_default_limits(self):
        """Joint barrier is expected to use limits from the model by default."""
        barrier = JointBarrier(
            name="test_barrier",
            gain=1.0,
        )
        with self.assertRaises(ValueError):
            _ = barrier.q_min

        with self.assertRaises(ValueError):
            _ = barrier.q_max

        barrier.update_model(self.model)
        np.testing.assert_array_equal(barrier.q_min, self.model.jnt_range[:, 0])
        np.testing.assert_array_equal(barrier.q_max, self.model.jnt_range[:, 1])

    def test_jax_component(self):
        barrier = JointBarrier(
            name="test_barrier",
            gain=1.0,
            q_min=[-1.0],
            q_max=[1.0],
        )
        barrier.update_model(self.model)

        jax_component = barrier.jax_component

        self.assertIsInstance(jax_component, JaxJointBarrier)
        self.assertEqual(jax_component.dim, 2)
        np.testing.assert_array_equal(jax_component.vector_gain, jnp.ones(2))
        np.testing.assert_array_equal(jax_component.full_q_min, jnp.array([-1.0]))
        np.testing.assert_array_equal(jax_component.full_q_max, jnp.array([1.0]))

    def test_call(self):
        """Test jax component actual computation"""
        barrier = JaxJointBarrier(
            dim=2,
            model=self.model,
            vector_gain=jnp.ones(2),
            gain_fn=lambda x: x,
            mask_idxs=(0,),
            safe_displacement_gain=0.0,
            full_q_min=jnp.array([-1.0]),
            full_q_max=jnp.array([1.0]),
            floating_base=False,
        )

        self.data.replace(qpos=jnp.array([0.5]))
        self.data = mjx.kinematics(self.model, self.data)

        result = barrier(self.data)
        expected = jnp.concatenate(
            [
                joint_difference(self.model, self.data.qpos, barrier.full_q_min),
                joint_difference(self.model, barrier.full_q_max, self.data.qpos),
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)
