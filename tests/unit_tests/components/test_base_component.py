import unittest

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components import Component, JaxComponent
from mjinx.typing import ArrayOrFloat


class DummyJaxComponent(JaxComponent):
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return data.qpos[: self.dim]  # Simple identity function for testing


class DummyComponent(Component[DummyJaxComponent]):
    JaxComponentType: type = DummyJaxComponent

    def define_dim(self, dim: int):
        self._dim = dim

    # def _build_component(self) -> DummyJaxComponent:
    #     return DummyJaxComponent(
    #         dim=self.dim,
    #         model=self.model,
    #         gain=self.gain,
    #         gain_function=self.gain_fn,
    #         mask_idxs=self.mask_idxs,
    #     )


class TestComponent(unittest.TestCase):
    dummy_dim: int = 3

    def setUp(self):
        """Setting up component test based on ComTask example.

        Note that ComTask has one of the smallest additional functionality and processing
        compared to other components"""

        self.component = DummyComponent("test_component", gain=2.0)

    def set_model(self, component: DummyComponent | None = None):
        """Setting the model to the component"""
        self.model = mjx.put_model(
            mj.MjModel.from_xml_string("<mujoco><worldbody><body name='body1'/></worldbody></mujoco>")
        )
        if component is None:
            self.component.update_model(self.model)
        else:
            component.update_model(self.model)

    def set_dim(self, component: DummyComponent | None = None):
        """Setting the model to the component"""
        if component is None:
            self.component.define_dim(self.dummy_dim)
        else:
            component.define_dim(self.dummy_dim)

    def set_gain_and_check(self, gain: ArrayOrFloat):
        """Checking that gain is transformed into jnp.ndarray, and values are correct"""
        self.component.update_gain(gain)
        np.testing.assert_array_equal(self.component.gain, gain)

    def test_update_gain(self):
        """Testing setting up different gains dimensions"""
        # Test scalar gain assignment
        self.set_gain_and_check(1.0)
        self.set_gain_and_check(np.array(2.0))
        self.set_gain_and_check(jnp.array(3.0))

        # Test vector gain assignment
        self.set_gain_and_check(np.zeros(self.dummy_dim))
        self.set_gain_and_check(jnp.ones(self.dummy_dim))

        # Test assigning gain with other dimension
        self.set_dim()
        gain = np.eye(self.component.dim)
        with self.assertRaises(ValueError):
            self.component.update_gain(gain)

    def test_vector_gain(self):
        """Testing proper convertation of gain to vector form"""
        # Vector gain could not be constucted till dimension is specified
        self.component.gain = 1.0
        with self.assertRaises(ValueError):
            _ = self.component.vector_gain

        self.set_dim()

        # Scalar -> jnp.ones(dim) * scalar
        self.component.update_gain(3.0)
        np.testing.assert_array_equal(self.component.vector_gain, jnp.ones(self.component.dim) * 3)

        # Vector -> vector
        vector_gain = jnp.arange(self.component.dim)
        self.component.update_gain(vector_gain)
        np.testing.assert_array_equal(self.component.vector_gain, vector_gain)

        # For vector gain, if the dimension turned out to be wrong, ValueError should be raised
        # Note that error would be raised only when vector_gain is accessed, even if model is already
        # provided
        vector_gain = jnp.ones(self.component.dim + 1)
        self.component.update_gain(vector_gain)
        with self.assertRaises(ValueError):
            _ = self.component.vector_gain

    def test_mask(self):
        """Testing output mask

        Note that length of the mask is not checked, because it's up to component to define how exactly
        to use the mask."""

        # Default mask
        # It should not be accessible till dimension is specified
        with self.assertRaises(ValueError):
            _ = self.component.mask
        with self.assertRaises(ValueError):
            _ = self.component.mask_idxs
        self.set_dim()
        # When dimension is specified, mask should be jnp.ones(dim)
        np.testing.assert_array_equal(self.component.mask, jnp.ones(self.component.dim))
        self.assertEqual(self.component.mask_idxs, tuple(range(self.component.dim)))

        # Default mask initialization
        # mask = [True, False, True, False, ...]
        mask_list = [True, False, True]
        mask_idxs_tuple = (0, 2)

        component_with_mask = DummyComponent("masked_component", gain=1.0, mask=mask_list)
        self.set_dim(component_with_mask)

        np.testing.assert_array_equal(component_with_mask.mask, jnp.array(mask_list))
        self.assertEqual(component_with_mask.mask_idxs, mask_idxs_tuple)

        # Wrong mask array ndim
        with self.assertRaises(ValueError):
            _ = DummyComponent("wrong_masked_component", gain=2.0, mask=[[True, False], [False, True]])

    def test_gain_function(self):
        """Testing that provided gain function is accessible and functions as expected"""

        def custom_gain_fn(x: float) -> float:
            return x**2

        component_with_gain_fn = DummyComponent("gain_fn_component", gain=2.0, gain_fn=custom_gain_fn)
        self.assertEqual(component_with_gain_fn.gain_fn(3.0), 9.0)

    def test_building_component(self):
        """Testing jax component build"""

        # First, model has to be provided
        with self.assertRaises(ValueError):
            _ = self.component.jax_component
        self.set_model()

        # Second, dimension has to be specified
        with self.assertRaises(ValueError):
            _ = self.component.jax_component
        self.set_dim()

        # Those conditions should always be enough to build jax component
        _ = self.component.jax_component

    def test_copy_and_set(self):
        """Test Component's method for copying and setting"""
        self.set_model()
        self.set_dim()
        jax_component = self.component.jax_component

        updated_jax_component = jax_component.copy_and_set(vector_gain=jnp.ones(self.component.dim))

        init_component_dict = jax_component.__dict__
        updated_component_dict = updated_jax_component.__dict__

        self.assertTrue(np.any(init_component_dict["vector_gain"] != updated_component_dict["vector_gain"]))
        init_component_dict.pop("vector_gain")
        updated_component_dict.pop("vector_gain")
        self.assertEqual(init_component_dict, updated_component_dict)

    def test_compute_jacobian(self):
        # Create a simple MuJoCo model with 3 degrees of freedom
        xml = """
        <mujoco>
          <worldbody>
            <body name="body1">
              <geom name="geom1" size=".1"/>
              <joint type="free"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mjx.put_model(mj.MjModel.from_xml_string(xml))
        data = mjx.make_data(model)

        # Initialize the TestJaxComponent
        component = DummyJaxComponent(
            dim=3, model=model, vector_gain=jnp.ones(3), gain_fn=lambda x: x, mask_idxs=(0, 1, 2)
        )

        # Set random qpos
        key = jax.random.PRNGKey(0)
        qpos = jax.random.uniform(key, (model.nq,))
        data = data.replace(qpos=qpos)

        # Compute the Jacobian
        jacobian = component.compute_jacobian(data)

        # Expected Jacobian for our simple identity function should be:
        # [1 0 0 0 0 0 0]
        # [0 1 0 0 0 0 0]
        # [0 0 1 0 0 0 0]
        expected_jacobian = jnp.eye(3, 6)

        # Check if the computed Jacobian matches the expected Jacobian
        np.testing.assert_allclose(jacobian, expected_jacobian, atol=1e-5)

        # Test that the Jacobian has the correct shape
        self.assertEqual(jacobian.shape, (component.dim, model.nv))

        # Test that the Jacobian is differentiable
        def jacobian_sum(qpos):
            data = mjx.make_data(model).replace(qpos=qpos)
            return jnp.sum(component.compute_jacobian(data))

        # Compute the gradient of the Jacobian sum
        gradient = jax.grad(jacobian_sum)(qpos)

        # Check that the gradient has the correct shape
        self.assertEqual(gradient.shape, (model.nq,))

        # Check that the gradient is zero (since our function is linear)
        np.testing.assert_allclose(gradient, jnp.zeros_like(gradient), atol=1e-5)
