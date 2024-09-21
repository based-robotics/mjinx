"""Center of mass task implementation."""

import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.tasks import JointTask


class TestJointTask(unittest.TestCase):
    def set_model(self, task: JointTask):
        self.model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
        <mujoco>
            <worldbody>
                <body name="body1">
                    <geom name="box1" size=".3"/>
                    <joint name="jnt1" type="hinge" axis="1 -1 0"/>
                    <body name="body2">
                        <joint name="jnt2" type="hinge" axis="1 -1 0"/>
                        <geom name="box2" pos=".6 .6 .6" size=".3"/>
                        <body name="body3">
                            <joint name="jnt3" type="hinge" axis="1 -1 0" pos=".6 .6 .6"/>
                            <geom name="box3" pos="1.2 1.2 1.2" size=".3"/>
                        </body>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """
            )
        )
        task.update_model(self.model)

    def test_set_model_no_mask(self):
        """Check the effect of setting the model for the task without mask"""
        jnt_task = JointTask("jnt_task", 1.0, 1.0)
        with self.assertRaises(ValueError):
            _ = jnt_task.dim
        with self.assertRaises(ValueError):
            _ = jnt_task.target_q

        self.set_model(jnt_task)

        self.assertEqual(jnt_task.dim, self.model.nq)
        np.testing.assert_array_equal(jnt_task.target_q, jnp.zeros(self.model.nq))
        jnt_task.update_target_q(jnp.ones(self.model.nq))
        with self.assertRaises(ValueError):
            jnt_task.update_target_q(jnp.ones(self.model.nq + 1))

    def test_set_model_with_mask(self):
        """Check the effect of setting the model for the task with mask"""
        jnt_task = JointTask("jnt_task", 1.0, 1.0, mask=[True, True, False])
        self.set_model(jnt_task)

        self.assertEqual(jnt_task.dim, 2)
        self.assertEqual(len(jnt_task.target_q), 2)

        jnt_task.update_target_q(jnp.ones(2))
        with self.assertRaises(ValueError):
            jnt_task.update_target_q(jnp.ones(self.model.nq))

        jnt_task_wrong_mask = JointTask("jnt_task", 1.0, 1.0, mask=[True, False])
        with self.assertRaises(ValueError):
            self.set_model(jnt_task_wrong_mask)

    def test_update_target_q(self):
        jnt_task = JointTask("jnt_task", 1.0, 1.0)
        new_target = [1.0, 2.0, 3.0]
        jnt_task.update_target_q(new_target)
        np.testing.assert_array_equal(jnt_task.target_q, new_target)

        # Setting wrong task target
        # Note that this could be detected only upon setting the model
        bad_target = [1.0, 2.0]
        jnt_task.update_target_q(bad_target)
        with self.assertRaises(ValueError):
            self.set_model(jnt_task)

        # Setting task target with mask
        jnt_task_masked = JointTask("jnt_task", 1.0, 1.0, mask=[True, True, False])
        jnt_task_masked.update_target_q(bad_target)
        self.set_model(jnt_task_masked)

    def test_build_component(self):
        jnt_task = JointTask(
            "jnt_task", cost=1.0, gain=2.0, gain_fn=lambda x: 2 * x, lm_damping=0.5, mask=[True, False, True]
        )
        self.set_model(jnt_task)
        jnt_des = jnp.array((-0.2, 0.4))
        jnt_task.target_q = jnt_des

        jax_component = jnt_task._build_component()
        self.assertEqual(jax_component.dim, 2)
        np.testing.assert_array_equal(jax_component.matrix_cost, jnp.eye(2))
        np.testing.assert_array_equal(jax_component.vector_gain, jnp.ones(2) * 2.0)
        self.assertEqual(jax_component.gain_fn(4), 8)
        self.assertEqual(jax_component.lm_damping, 0.5)
        np.testing.assert_array_equal(jax_component.target_q, jnt_des)
        self.assertEqual(jax_component.mask_idxs, (0, 2))

        data = mjx.fwd_position(self.model, mjx.make_data(self.model))
        com_value = jax_component(data)
        np.testing.assert_array_equal(com_value, jnp.array([0.2, -0.4]))
