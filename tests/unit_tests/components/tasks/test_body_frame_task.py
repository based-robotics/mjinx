"""Center of mass task implementation."""

import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3

from mjinx.components.tasks import FrameTask


class TestBodyFrameTask(unittest.TestCase):

    def setUp(self):
        self.to_wxyz_xyz: jnp.array = jnp.array([3, 4, 5, 6, 0, 1, 2])

    def set_model(self, task: FrameTask):
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
        frame_task = FrameTask("frame_task", cost=1.0, gain=2.0, body_name="body3")
        # By default, it has to be identity
        np.testing.assert_array_almost_equal(SE3.identity().wxyz_xyz, frame_task.target_frame.wxyz_xyz)

        # Setting with SE3 object
        test_se3 = SE3(jnp.array([0.1, 0.2, -0.1, 0, 1, 0, 0]))
        frame_task.target_frame = test_se3
        np.testing.assert_array_almost_equal(test_se3.wxyz_xyz, frame_task.target_frame.wxyz_xyz)

        # Setting with sequence
        test_se3_seq = (-1, 0, 1, 0, 0, 1, 0)
        frame_task.target_frame = test_se3_seq
        np.testing.assert_array_almost_equal(
            jnp.array(test_se3_seq)[self.to_wxyz_xyz], frame_task.target_frame.wxyz_xyz
        )

    def test_build_component(self):
        frame_task = FrameTask(
            "frame_task",
            cost=1.0,
            gain=2.0,
            body_name="body3",
            gain_fn=lambda x: 2 * x,
            lm_damping=0.5,
            mask=[True, False, True, True, False, True],
        )
        self.set_model(frame_task)
        frame_des = jnp.array([0.1, 0, -0.1, 1, 0, 0, 0])
        frame_task.target_frame = frame_des

        jax_component = frame_task.jax_component

        self.assertEqual(jax_component.dim, 4)
        np.testing.assert_array_equal(jax_component.cost, jnp.eye(jax_component.dim))
        np.testing.assert_array_equal(jax_component.gain, jnp.ones(jax_component.dim) * 2.0)
        np.testing.assert_array_equal(jax_component.body_id, frame_task.body_id)
        self.assertEqual(jax_component.gain_function(4), 8)
        self.assertEqual(jax_component.lm_damping, 0.5)
        np.testing.assert_array_almost_equal(jax_component.target_frame.wxyz_xyz, frame_des[self.to_wxyz_xyz])

        self.assertEqual(jax_component.mask_idxs, (0, 2, 3, 5))

        data = mjx.fwd_position(self.model, mjx.make_data(self.model))
        error = jax_component(data)
        np.testing.assert_array_equal(error, jnp.array([0.1, -0.1, 0.0, 0.0]))

        # Testing component jacobian
        jac = jax_component.compute_jacobian(mjx.make_data(self.model))

        self.assertEqual(jac.shape, (jax_component.dim, self.model.nv))


unittest.main()
