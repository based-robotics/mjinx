import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.tasks import BodyTask, JaxBodyTask


class DummyJaxBodyTask(JaxBodyTask):
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return data.xpos[self.body_id]


class DummyBodyTask(BodyTask[DummyJaxBodyTask]):
    def set_dim(self, dim: int):
        self._dim = dim

    def _build_component(self) -> DummyJaxBodyTask:
        return DummyJaxBodyTask(
            dim=self.dim,
            model=self.model,
            gain=self.gain,
            gain_function=self.gain_fn,
            mask_idxs=self.mask_idxs,
            cost=self.cost,
            lm_damping=self.lm_damping,
            body_id=self.body_id,
        )


class TestBodyTask(unittest.TestCase):
    def set_model(self, task: DummyBodyTask):
        task.update_model(
            mjx.put_model(
                mj.MjModel.from_xml_string(
                    "<mujoco><worldbody><body name='body1'/> <body name='body2'/></worldbody></mujoco>"
                )
            )
        )

    def test_body_id(self):
        body_task_1 = DummyBodyTask("body_task", cost=1.0, gain=1.0, body_name="body1")
        self.set_model(body_task_1)

        self.assertEqual(body_task_1.body_id, 1)

        body_task_2 = DummyBodyTask("body_task", cost=1.0, gain=1.0, body_name="body2")
        self.set_model(body_task_2)

        self.assertEqual(body_task_2.body_id, 2)

    def test_update_model_invalid_body(self):
        body_task = DummyBodyTask("body_task", cost=1.0, gain=1.0, body_name="body3")
        with self.assertRaises(ValueError):
            self.set_model(body_task)
