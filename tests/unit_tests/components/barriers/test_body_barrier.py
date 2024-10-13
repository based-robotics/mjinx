import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.barriers import BodyBarrier, JaxBodyBarrier


class DummyJaxBodyBarrier(JaxBodyBarrier):
    """Dummy class to make minimal non-abstract jax body barrier class"""

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return data.xpos[self.body_id]


class DummyBodyTask(BodyBarrier[JaxBodyBarrier]):
    """Dummy class to make minimal non-abstract jax body barrier class"""

    def set_dim(self, dim: int):
        self._dim = dim


class TestBodyTask(unittest.TestCase):
    def set_model(self, task: DummyBodyTask):
        """Helper class to set the model"""
        self.mj_model = mj.MjModel.from_xml_string(
            "<mujoco><worldbody><body name='body1'/> <body name='body2'/></worldbody></mujoco>"
        )
        self.mjx_model = mjx.put_model(self.mj_model)
        task.update_model(self.mjx_model)

    def test_body_id(self):
        """Test setting body id"""
        body_task_1 = DummyBodyTask("body_task", gain=1.0, body_name="body1")
        with self.assertRaises(ValueError):
            _ = body_task_1.body_id
        self.set_model(body_task_1)

        self.assertEqual(body_task_1.body_id, 1)
        self.assertEqual(body_task_1.body_name, mj.mj_id2name(self.mj_model, mj.mjtObj.mjOBJ_BODY, body_task_1.body_id))

        body_task_2 = DummyBodyTask("body_task", gain=1.0, body_name="body2")
        self.set_model(body_task_2)

        self.assertEqual(body_task_2.body_id, 2)
        self.assertEqual(body_task_2.body_name, mj.mj_id2name(self.mj_model, mj.mjtObj.mjOBJ_BODY, body_task_2.body_id))

    def test_update_model_invalid_body(self):
        """Test trying to set invalid body as desired one"""
        body_task = DummyBodyTask("body_task", gain=1.0, body_name="body3")
        with self.assertRaises(ValueError):
            self.set_model(body_task)
