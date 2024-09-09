import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3

from mjinx.components.tasks import FrameTask


class TestFrameTask(unittest.TestCase):
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

    def setUp(self):
        self.frame_task = FrameTask("frame_task", cost=1.0, gain=1.0, body_name="body1")

    def test_task_dim(self):
        self.assertEqual(self.frame_task.dim, 6)

    def test_update_target_frame_from_SE3(self):
        new_target = SE3.from_rotation_and_translation(SO3.identity(), jnp.array([1.0, 2.0, 3.0]))
        self.frame_task.update_target_frame(new_target)
        np.testing.assert_array_almost_equal(self.frame_task.target_frame.wxyz_xyz, new_target.wxyz_xyz)

    def test_update_target_frame_from_sequence(self):
        new_target = [0, 0, 0, 1.0, 0, 0, 0]
        new_target_se3 = SE3(np.array(new_target)[[3, 4, 5, 6, 0, 1, 2]])
        self.frame_task.update_target_frame(new_target)
        np.testing.assert_array_almost_equal(self.frame_task.target_frame.wxyz_xyz, new_target_se3.wxyz_xyz)

    def test_mask_handling(self):
        masked_frame_task = FrameTask(
            "masked_frame_task", cost=1.0, gain=1.0, body_name="body1", mask=[0, 0, 0, 1, 1, 1]
        )

        self.assertEqual(masked_frame_task.dim, 3)

    def test_build_component(self):
        frame_task = FrameTask(
            "com_task",
            cost=1.0,
            gain=2.0,
            gain_fn=lambda x: 2 * x,
            lm_damping=0.5,
            mask=[0, 0, 0, 1, 1, 1],
            body_name="body1",
        )
        self.set_model(frame_task)
        frame_des = jnp.array([0, 0, 0, 1, 0, 0, 0])
        frame_task.target_frame = frame_des

        jax_component = frame_task._build_component()
        self.assertEqual(jax_component.dim, 3)
        np.testing.assert_array_equal(jax_component.cost, jnp.eye(3))
        np.testing.assert_array_equal(jax_component.gain, jnp.ones(3) * 2.0)
        self.assertEqual(jax_component.gain_function(4), 8)
        self.assertEqual(jax_component.lm_damping, 0.5)
        np.testing.assert_array_almost_equal(jax_component.target_frame.wxyz_xyz, frame_des[(3, 4, 5, 6, 0, 1, 2),])
        self.assertEqual(jax_component.mask_idxs, (3, 4, 5))

        data = mjx.fwd_position(self.model, mjx.make_data(self.model))
        com_value = jax_component(data)
        np.testing.assert_array_almost_equal(com_value, jnp.zeros(3))


unittest.main()
