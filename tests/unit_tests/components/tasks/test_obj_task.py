import unittest
from collections.abc import Callable, Sequence

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3

from mjinx.components.tasks import JaxObjTask, ObjTask
from mjinx.typing import ArrayOrFloat


class DummyJaxObjTask(JaxObjTask):
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return data.xpos[self.obj_id]


class DummyObjTask(ObjTask[DummyJaxObjTask]):
    JaxComponentType = DummyJaxObjTask

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        obj_name: str,
        obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, cost, gain, obj_name, obj_type, gain_fn, lm_damping, mask)
        self._dim = 3


class TestObjTask(unittest.TestCase):
    def set_model(self, task: DummyObjTask):
        self.mj_model = mj.MjModel.from_xml_string(
            "<mujoco>"
            "<worldbody>"
            "<body name='body1' pos='1 2 3' quat='1 0 0 0'>"
            "<geom name='geom1' pos='7 8 9' quat='0 0 1 1' size='0.1'/>"
            "<joint/>"
            "</body>"
            "<site name='site1' pos='10 11 12' quat='1 1 0 0'/>"
            "</worldbody>"
            "</mujoco>"
        )
        self.mjx_model = mjx.put_model(self.mj_model)
        task.update_model(self.mjx_model)
        self.mjx_data = mjx.make_data(self.mjx_model)
        self.mjx_data = mjx.kinematics(self.mjx_model, self.mjx_data)

    def test_obj_id(self):
        # Test body
        obj_task_body = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="body1", obj_type=mj.mjtObj.mjOBJ_BODY)
        self.set_model(obj_task_body)

        self.assertEqual(obj_task_body.obj_id, 1)
        self.assertEqual(obj_task_body.obj_name, "body1")
        self.assertEqual(obj_task_body.obj_type, mj.mjtObj.mjOBJ_BODY)

        # Test geom
        obj_task_geom = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="geom1", obj_type=mj.mjtObj.mjOBJ_GEOM)
        self.set_model(obj_task_geom)

        self.assertEqual(obj_task_geom.obj_id, 0)
        self.assertEqual(obj_task_geom.obj_name, "geom1")
        self.assertEqual(obj_task_geom.obj_type, mj.mjtObj.mjOBJ_GEOM)

        # Test site
        obj_task_site = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="site1", obj_type=mj.mjtObj.mjOBJ_SITE)
        self.set_model(obj_task_site)

        self.assertEqual(obj_task_site.obj_id, 0)
        self.assertEqual(obj_task_site.obj_name, "site1")
        self.assertEqual(obj_task_site.obj_type, mj.mjtObj.mjOBJ_SITE)

    def test_update_model_invalid_object(self):
        obj_task = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="invalid_obj", obj_type=mj.mjtObj.mjOBJ_BODY)
        with self.assertRaises(ValueError):
            self.set_model(obj_task)

    def test_default_obj_type(self):
        obj_task = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="body1")
        self.set_model(obj_task)
        self.assertEqual(obj_task.obj_type, mj.mjtObj.mjOBJ_BODY)

    def test_get_pos(self):
        # Test for body
        obj_task_body = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="body1", obj_type=mj.mjtObj.mjOBJ_BODY)
        self.set_model(obj_task_body)
        jax_task = obj_task_body.jax_component
        pos = jax_task.get_pos(self.mjx_data)
        np.testing.assert_allclose(pos, np.array([1.0, 2.0, 3.0]), rtol=1e-4, atol=1e-4)

        # Test for geom
        obj_task_geom = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="geom1", obj_type=mj.mjtObj.mjOBJ_GEOM)
        self.set_model(obj_task_geom)
        jax_task = obj_task_geom.jax_component
        pos = jax_task.get_pos(self.mjx_data)
        np.testing.assert_allclose(pos, np.array([7.0 + 1.0, 8.0 + 2.0, 9.0 + 3.0]), rtol=1e-4, atol=1e-4)

        # Test for site
        obj_task_site = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="site1", obj_type=mj.mjtObj.mjOBJ_SITE)
        self.set_model(obj_task_site)
        jax_task = obj_task_site.jax_component
        pos = jax_task.get_pos(self.mjx_data)
        np.testing.assert_allclose(pos, np.array([10.0, 11.0, 12.0]), rtol=1e-4, atol=1e-4)

    def test_get_rotation(self):
        # Test for body
        obj_task_body = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="body1", obj_type=mj.mjtObj.mjOBJ_BODY)
        self.set_model(obj_task_body)
        jax_task = obj_task_body.jax_component
        rotation = jax_task.get_rotation(self.mjx_data)
        expected_rotation = SO3.from_quaternion_xyzw(jnp.array([0.0, 0.0, 0.0, 1.0]))
        np.testing.assert_allclose(rotation.as_matrix(), expected_rotation.as_matrix(), rtol=1e-4, atol=1e-4)

        # Test for geom
        obj_task_geom = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="geom1", obj_type=mj.mjtObj.mjOBJ_GEOM)
        self.set_model(obj_task_geom)
        jax_task = obj_task_geom.jax_component
        rotation = jax_task.get_rotation(self.mjx_data)
        expected_rotation = SO3.from_quaternion_xyzw(jnp.array([0.0, 0.7071068, 0.7071068, 0.0]))
        np.testing.assert_allclose(rotation.as_matrix(), expected_rotation.as_matrix(), rtol=1e-4, atol=1e-4)

        # Test for site
        obj_task_site = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="site1", obj_type=mj.mjtObj.mjOBJ_SITE)
        self.set_model(obj_task_site)
        jax_task = obj_task_site.jax_component
        rotation = jax_task.get_rotation(self.mjx_data)
        expected_rotation = SO3.from_quaternion_xyzw(jnp.array([0.7071068, 0.0, 0.0, 0.7071068]))
        np.testing.assert_allclose(rotation.as_matrix(), expected_rotation.as_matrix(), rtol=1e-4, atol=1e-4)

    def test_get_frame(self):
        # Test for body
        obj_task_body = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="body1", obj_type=mj.mjtObj.mjOBJ_BODY)
        self.set_model(obj_task_body)
        jax_task = obj_task_body.jax_component
        frame = jax_task.get_frame(self.mjx_data)
        expected_frame = SE3.from_rotation_and_translation(
            SO3.from_quaternion_xyzw(jnp.array([0.0, 0.0, 0.0, 1.0])), jnp.array([1.0, 2.0, 3.0])
        )
        np.testing.assert_allclose(frame.as_matrix(), expected_frame.as_matrix(), rtol=1e-4, atol=1e-4)

        # Test for geom
        obj_task_geom = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="geom1", obj_type=mj.mjtObj.mjOBJ_GEOM)
        self.set_model(obj_task_geom)
        jax_task = obj_task_geom.jax_component
        frame = jax_task.get_frame(self.mjx_data)
        expected_frame = SE3.from_rotation_and_translation(
            SO3.from_quaternion_xyzw(jnp.array([0.0, 0.7071068, 0.7071068, 0.0])),
            jnp.array([7.0 + 1.0, 8.0 + 2.0, 9.0 + 3.0]),
        )
        np.testing.assert_allclose(frame.as_matrix(), expected_frame.as_matrix(), rtol=1e-4, atol=1e-4)

        # Test for site
        obj_task_site = DummyObjTask("obj_task", cost=1.0, gain=1.0, obj_name="site1", obj_type=mj.mjtObj.mjOBJ_SITE)
        self.set_model(obj_task_site)
        jax_task = obj_task_site.jax_component
        frame = jax_task.get_frame(self.mjx_data)
        expected_frame = SE3.from_rotation_and_translation(
            SO3.from_quaternion_xyzw(jnp.array([0.7071068, 0.0, 0.0, 0.7071068])), jnp.array([10.0, 11.0, 12.0])
        )
        np.testing.assert_allclose(frame.as_matrix(), expected_frame.as_matrix(), rtol=1e-4, atol=1e-4)
