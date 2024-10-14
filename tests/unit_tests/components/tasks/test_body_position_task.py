"""Center of mass task implementation."""

import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.tasks import PositionTask


class TestBodyPositionTask(unittest.TestCase):
    def set_model(self, task: PositionTask):
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

    def test_target_frame(self):
        """Testing manipulations with target frame"""
        pos_task = PositionTask("pos_task", cost=1.0, gain=2.0, obj_name="body3")
        # By default, it has to be identity
        np.testing.assert_array_almost_equal(pos_task.target_pos, jnp.zeros(3))

        # Setting with sequence
        test_pos = (-1, 0, 1)
        pos_task.target_pos = test_pos
        np.testing.assert_array_almost_equal(jnp.array(test_pos), pos_task.target_pos)

        with self.assertRaises(ValueError):
            pos_task.target_pos = (0, 1, 2, 3, 4)

    def test_build_component(self):
        frame_task = PositionTask(
            "frame_task",
            cost=1.0,
            gain=2.0,
            obj_name="body3",
            gain_fn=lambda x: 2 * x,
            lm_damping=0.5,
            mask=[True, False, True],
        )
        self.set_model(frame_task)
        pos_des = jnp.array([0.1, -0.1])
        frame_task.target_pos = pos_des

        jax_component = frame_task.jax_component

        self.assertEqual(jax_component.dim, 2)
        np.testing.assert_array_equal(jax_component.matrix_cost, jnp.eye(jax_component.dim))
        np.testing.assert_array_equal(jax_component.vector_gain, jnp.ones(jax_component.dim) * 2.0)
        np.testing.assert_array_equal(jax_component.obj_id, frame_task.obj_id)
        self.assertEqual(jax_component.gain_fn(4), 8)
        self.assertEqual(jax_component.lm_damping, 0.5)
        np.testing.assert_array_almost_equal(jax_component.target_pos, pos_des)

        self.assertEqual(jax_component.mask_idxs, (0, 2))

        data = mjx.fwd_position(self.model, mjx.make_data(self.model))
        error = jax_component(data)
        np.testing.assert_array_equal(error, jnp.array([-0.1, 0.1]))
