import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3

from mjinx.components.tasks import PositionTask


class TestPositionTask(unittest.TestCase):
    def set_model(self, task: PositionTask):
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

    def setUp(self):
        self.pos_task = PositionTask("pos_task", cost=1.0, gain=1.0, body_name="body1")

    def test_task_dim(self):
        self.assertEqual(self.pos_task.dim, 3)

    def test_update_target_pos(self):
        new_target = [0, 1, 2]
        self.pos_task.update_target_pos(new_target)
        np.testing.assert_array_almost_equal(self.pos_task.target_pos, jnp.array(new_target))

    def test_mask_handling(self):
        masked_pos_task = PositionTask("masked_pos_task", cost=1.0, gain=1.0, body_name="body1", mask=[0, 0, 1])

        self.assertEqual(masked_pos_task.dim, 1)

    def test_update_target_pos_with_mask(self):
        masked_pos_task = PositionTask("masked_pos_task", cost=1.0, gain=1.0, body_name="body1", mask=[0, 0, 1])

        with self.assertRaises(ValueError):
            masked_pos_task.target_pos = [0, 1, 2]
        masked_pos_task.update_target_pos([1])
        self.assertEqual(len(masked_pos_task.target_pos), 1)

    def test_build_component(self):
        pos_task = PositionTask(
            "pos_task",
            cost=1.0,
            gain=2.0,
            gain_fn=lambda x: 2 * x,
            lm_damping=0.5,
            mask=[0, 1, 0],
            body_name="body1",
        )
        self.set_model(pos_task)
        pos_des = jnp.array([1])
        pos_task.target_pos = pos_des

        jax_component = pos_task._build_component()
        self.assertEqual(jax_component.dim, 1)
        np.testing.assert_array_equal(jax_component.cost, jnp.eye(1))
        np.testing.assert_array_equal(jax_component.gain, jnp.ones(1) * 2.0)
        self.assertEqual(jax_component.gain_function(4), 8)
        self.assertEqual(jax_component.lm_damping, 0.5)
        np.testing.assert_array_almost_equal(jax_component.target_pos, pos_des)
        self.assertEqual(jax_component.mask_idxs, (1,))

        data = mjx.fwd_position(self.model, mjx.make_data(self.model))
        com_value = jax_component(data)
        np.testing.assert_array_almost_equal(com_value, -1 * jnp.ones(1))


unittest.main()
