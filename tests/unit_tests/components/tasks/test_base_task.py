import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.tasks import JaxTask, Task
from mjinx.typing import ArrayOrFloat


class DummyJaxTask(JaxTask):
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        """Immitating CoM task"""
        return data.subtree_com[self.model.body_rootid[0], self.mask_idxs]


class DummyTask(Task[DummyJaxTask]):
    JaxComponentType: type = DummyJaxTask

    def define_dim(self, dim: int):
        self._dim = dim


class TestTask(unittest.TestCase):
    dummy_dim: int = 3

    def setUp(self):
        self.model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
        <mujoco>
            <worldbody>
                <body name="body1">
                    <geom name="box1" size=".3"/>
                    <joint name="jnt1" type="hinge" axis="1 -1 0"/>
                    <body name="body2"/>
                </body>
            </worldbody>
        </mujoco>
        """
            )
        )
        self.task = DummyTask("test_task", cost=2.0, gain=1.0)
        self.task.update_model(self.model)

    def set_dim(self):
        self.task.define_dim(self.dummy_dim)

    def __set_cost_and_check(self, cost: ArrayOrFloat):
        """Checking that cost is transformed into jnp.ndarray, and values are correct"""
        self.task.update_cost(cost)
        np.testing.assert_array_equal(self.task.cost, cost)

    def test_update_gain(self):
        """Testing setting up different cost dimensions"""
        # Test scalar cost assignment
        self.__set_cost_and_check(1.0)
        self.__set_cost_and_check(np.array(2.0))
        self.__set_cost_and_check(jnp.array(3.0))

        # Test vector cost assignment
        self.__set_cost_and_check(np.zeros(self.dummy_dim))
        self.__set_cost_and_check(jnp.ones(self.dummy_dim))

        # Test matrix cost assignment
        self.__set_cost_and_check(np.eye(self.dummy_dim))
        self.__set_cost_and_check(2 * jnp.eye(self.dummy_dim))

        # Test assigning cost with other dimension
        cost = np.zeros((self.dummy_dim, self.dummy_dim, self.dummy_dim))
        with self.assertRaises(ValueError):
            self.task.update_cost(cost)

    def test_matrix_cost(self):
        """Testing proper convertation of cost to matrix form"""
        # Matrix could not be constucted till dimension is specified
        self.task.cost = 1.0
        with self.assertRaises(ValueError):
            _ = self.task.matrix_cost

        self.set_dim()

        # Scalar -> jnp.eye(dim) * scalar
        self.task.update_cost(3.0)
        np.testing.assert_array_equal(self.task.matrix_cost, jnp.eye(self.task.dim) * 3)

        # Vector -> jnp.diag(vector)
        vector_cost = jnp.arange(self.task.dim)
        self.task.update_cost(vector_cost)
        np.testing.assert_array_equal(self.task.matrix_cost, jnp.diag(vector_cost))

        # Matrix -> matrix
        matrix_cost = jnp.eye(self.task.dim)
        self.task.update_cost(matrix_cost)
        np.testing.assert_array_equal(self.task.matrix_cost, matrix_cost)

        # For vector cost, the length has to be equal to dimension of the task
        vector_cost = jnp.ones(self.task.dim + 1)
        self.task.update_cost(vector_cost)
        with self.assertRaises(ValueError):
            _ = self.task.matrix_cost

        # For matrix cost, if the dimension turned out to be wrong, ValueError should be raised
        # Note that error would be raised only when matrix_cost is accessed, even if model is already
        # provided
        matrix_cost = jnp.eye(self.task.dim + 1)
        self.task.update_cost(matrix_cost)
        with self.assertRaises(ValueError):
            _ = self.task.matrix_cost

    def test_lm_damping(self):
        lm_specific_task = DummyTask("lm_specific_task", 1.0, 1.0, lm_damping=5.0)
        self.assertEqual(lm_specific_task.lm_damping, 5.0)

        with self.assertRaises(ValueError):
            _ = DummyTask("negative_lm_task", 1.0, 1.0, lm_damping=-5.0)

    def test_error(self):
        self.set_dim()
        jax_task = self.task.jax_component
        data = mjx.fwd_position(self.model, mjx.make_data(self.model))
        np.testing.assert_array_almost_equal(jax_task(data), jax_task.compute_error(data))
