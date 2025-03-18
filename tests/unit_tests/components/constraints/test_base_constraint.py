import unittest
import numpy as np
from mjinx.components.constraints import Constraint, JaxConstraint

# Language: python
import jax.numpy as jnp


class DummyConstraint(Constraint):
    dim = 3
    pass


class TestJaxConstraint(unittest.TestCase):
    def setUp(self):
        # Use a fixed dimension for tests.
        self.dim = 3

    def create_constraint(self, soft_cost):
        # Create a dummy constraint instance and set the required dim attribute.
        instance = DummyConstraint(name="dummy", gain=1.0, soft_constraint_cost=soft_cost)
        return instance

    def test_soft_constraint_cost_none(self):
        # Test when soft_constraint_cost is None, property returns 1e2 * identity matrix.
        constraint = self.create_constraint(None)
        expected = 1e2 * jnp.eye(self.dim)
        self.assertTrue(jnp.allclose(constraint.soft_constraint_cost, expected))

    def test_soft_constraint_cost_scalar(self):
        # Test when soft_constraint_cost is a scalar.
        scalar_value = 10.0
        constraint = self.create_constraint(scalar_value)
        expected = jnp.eye(self.dim) * scalar_value
        self.assertTrue(jnp.allclose(constraint.soft_constraint_cost, expected))

    def test_soft_constraint_cost_vector_correct(self):
        # Test when soft_constraint_cost is a vector with proper length.
        vector_cost = jnp.array([1.0, 2.0, 3.0])
        constraint = self.create_constraint(vector_cost)
        expected = jnp.diag(vector_cost)
        self.assertTrue(jnp.allclose(constraint.soft_constraint_cost, expected))

    def test_soft_constraint_cost_vector_invalid(self):
        # Test when soft_constraint_cost is a vector with incorrect length.
        vector_cost = jnp.array([1.0, 2.0])  # length != self.dim
        constraint = self.create_constraint(vector_cost)
        with self.assertRaises(ValueError):
            _ = constraint.soft_constraint_cost

    def test_soft_constraint_cost_matrix_correct(self):
        # Test when soft_constraint_cost is a matrix of proper shape.
        matrix_cost = jnp.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
        constraint = self.create_constraint(matrix_cost)
        self.assertTrue(jnp.allclose(constraint.soft_constraint_cost, matrix_cost))

    def test_soft_constraint_cost_matrix_invalid(self):
        # Test when soft_constraint_cost is a matrix with an incorrect shape.
        matrix_cost = jnp.array([[1.0, 0], [0, 2.0]])
        constraint = self.create_constraint(matrix_cost)
        with self.assertRaises(ValueError):
            _ = constraint.soft_constraint_cost

    def test_soft_constraint_cost_ndim_invalid(self):
        # Test when soft_constraint_cost has ndim > 2.
        invalid_cost = jnp.ones((self.dim, self.dim, 1))
        constraint = self.create_constraint(invalid_cost)
        with self.assertRaises(ValueError):
            _ = constraint.soft_constraint_cost


if __name__ == "__main__":
    unittest.main()
