import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.tasks import ComTask


class TestComTask(unittest.TestCase):
    def set_model(self, task: ComTask):
        self.model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
        <mujoco>
            <worldbody>
                <body name="body1">
                    <joint name="jnt1" type="hinge" axis="1 -1 0"/>
                    <geom name="box1" size=".3"/>
                    <geom name="box2" pos=".6 .6 .6" size=".3"/>
                </body>
            </worldbody>w
        </mujoco>
        """
            )
        )
        task.update_model(self.model)

    def test_task_dim(self):
        """Test task dimensionality"""
        com_task_masked = ComTask("com_task_masked", cost=1.0, gain=1.0, mask=[True, False, True])
        self.assertEqual(com_task_masked.dim, 2)

    def test_mask_validation(self):
        """Test task's mask size validation"""
        with self.assertRaises(ValueError):
            _ = ComTask("com_task_small_mask", cost=1.0, gain=1.0, mask=[True, False])

        with self.assertRaises(ValueError):
            _ = ComTask("com_task_big_mask", cost=1.0, gain=1.0, mask=[True, False, True, False])

    def test_update_target_com(self):
        """Test target CoM parameter updates"""
        com_task = ComTask("com_task_default", cost=1.0, gain=1.0)
        new_target = [1.0, 2.0, 3.0]
        com_task.update_target_com(new_target)
        np.testing.assert_array_equal(com_task.target_com, new_target)

        too_big_target = range(4)
        with self.assertRaises(ValueError):
            com_task.update_target_com(too_big_target)

    def test_build_component(self):
        com_task = ComTask(
            "com_task", cost=1.0, gain=2.0, gain_fn=lambda x: 2 * x, lm_damping=0.5, mask=[True, True, False]
        )
        self.set_model(com_task)
        com_des = jnp.array((-0.3, 0.3))
        com_task.target_com = com_des

        jax_component = com_task._build_component()
        self.assertEqual(jax_component.dim, 2)
        np.testing.assert_array_equal(jax_component.matrix_cost, jnp.eye(2))
        np.testing.assert_array_equal(jax_component.vector_gain, jnp.ones(2) * 2.0)
        self.assertEqual(jax_component.gain_fn(4), 8)
        self.assertEqual(jax_component.lm_damping, 0.5)
        np.testing.assert_array_equal(jax_component.target_com, com_des)
        self.assertEqual(jax_component.mask_idxs, (0, 1))

        data = mjx.fwd_position(self.model, mjx.make_data(self.model))
        com_value = jax_component(data)
        np.testing.assert_array_equal(com_value, jnp.array([0.6, 0.0]))
