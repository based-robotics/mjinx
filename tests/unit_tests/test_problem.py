import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.barriers import JointBarrier
from mjinx.components.tasks import ComTask, JaxComTask
from mjinx.problem import Problem


class TestProblem(unittest.TestCase):
    dummy_dim: int = 3

    def setUp(self):
        """Setting up component test based on ComTask example.

        Note that ComTask has one of the smallest additional functionality and processing
        compared to other components"""

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
        self.problem = Problem(
            self.model,
            v_min=0,
            v_max=0,
        )

    def test_v_min(self):
        """Testing proper v lower limit"""

        self.problem.v_min = 1

        self.assertIsInstance(self.problem.v_min, jnp.ndarray)
        np.testing.assert_array_equal(self.problem.v_min, jnp.ones(self.model.nv))

        with self.assertRaises(ValueError):
            self.problem.v_min = jnp.array([1, 2, 3])

        self.problem.v_min = np.arange(self.model.nv)
        self.assertIsInstance(self.problem.v_min, jnp.ndarray)
        np.testing.assert_array_equal(self.problem.v_min, jnp.arange(self.model.nv))

        jax_problem_data = self.problem.compile()
        self.assertIsInstance(jax_problem_data.v_min, jnp.ndarray)
        np.testing.assert_array_equal(jax_problem_data.v_min, jnp.arange(self.model.nv))

        with self.assertRaises(ValueError):
            self.problem.v_min = jnp.eye(self.model.nv)

    def test_v_max(self):
        """Testing proper v upper limit"""

        self.problem.v_max = 1

        np.testing.assert_array_equal(self.problem.v_max, jnp.ones(self.model.nv))

        with self.assertRaises(ValueError):
            self.problem.v_max = jnp.array([1, 2, 3])

        self.problem.v_max = np.arange(self.model.nv)
        self.assertIsInstance(self.problem.v_max, jnp.ndarray)
        np.testing.assert_array_equal(self.problem.v_max, jnp.arange(self.model.nv))

        jax_problem_data = self.problem.compile()
        self.assertIsInstance(jax_problem_data.v_max, jnp.ndarray)
        np.testing.assert_array_equal(jax_problem_data.v_max, jnp.arange(self.model.nv))

        with self.assertRaises(ValueError):
            self.problem.v_max = jnp.eye(self.model.nv)

    def test_add_component(self):
        """Testing adding component"""

        component = ComTask("test_component", 1.0, 1.0)

        self.problem.add_component(component)
        # Chech that model is defined and does not raise errors
        _ = component.model

        jax_problem_data = self.problem.compile()

        # False -> component was compiled as well
        self.assertFalse(component._modified)
        self.assertEqual(len(jax_problem_data.components), 1)
        self.assertIsInstance(jax_problem_data.components["test_component"], JaxComTask)

        component_with_same_name = JointBarrier("test_component", 1.0)

        with self.assertRaises(ValueError):
            self.problem.add_component(component_with_same_name)

    def test_remove_component(self):
        """Testing removing component"""
        component = ComTask("test_component", 1.0, 1.0)

        self.problem.add_component(component)
        self.problem.remove_component(component.name)

        jax_problem_data = self.problem.compile()
        self.assertEqual(len(jax_problem_data.components), 0)

    def test_components_access(self):
        """Test componnets access interface"""

        component = ComTask("test_component", 1.0, 1.0)

        self.problem.add_component(component)

        self.assertIsInstance(self.problem.component("test_component"), ComTask)

        with self.assertRaises(ValueError):
            _ = self.problem.component("non_existens_component")

    def test_tasks_access(self):
        """Test tasks access interface"""

        task = ComTask("test_task", 1.0, 1.0)
        barrier = JointBarrier("test_barrier", 1.0)

        self.problem.add_component(task)
        self.problem.add_component(barrier)

        self.assertIsInstance(self.problem.task("test_task"), ComTask)

        with self.assertRaises(ValueError):
            _ = self.problem.task("test_barrier")

        with self.assertRaises(ValueError):
            _ = self.problem.task("non_existent_task")

    def test_barriers_access(self):
        """Test barriers access interface"""

        task = ComTask("test_task", 1.0, 1.0)
        barrier = JointBarrier("test_barrier", 1.0)

        self.problem.add_component(task)
        self.problem.add_component(barrier)

        self.assertIsInstance(self.problem.barrier("test_barrier"), JointBarrier)

        with self.assertRaises(ValueError):
            _ = self.problem.barrier("test_task")

        with self.assertRaises(ValueError):
            _ = self.problem.barrier("non_existent_task")

    def test_setting_vmap_dimension(self):
        """Testing context manager for vmapping the dimensions"""

        task = ComTask("test_task", 1.0, 1.0)
        barrier = JointBarrier("test_barrier", 1.0)

        self.problem.add_component(task)
        self.problem.add_component(barrier)

        with self.problem.set_vmap_dimension() as empty_problem_data:
            empty_problem_data.components["test_task"].target_com = 0

        self.assertEqual(empty_problem_data.v_min, None)
        self.assertEqual(empty_problem_data.v_max, None)

        self.assertEqual(empty_problem_data.components["test_task"].target_com, 0)
