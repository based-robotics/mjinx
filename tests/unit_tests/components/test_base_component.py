import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components import Component, JaxComponent
from mjinx.typing import ArrayOrFloat


class DummyJaxComponent(JaxComponent):
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return data.subtree_com[self.model.body_rootid[0], self.mask_idxs]


class DummyComponent(Component[DummyJaxComponent]):
    def define_dim(self, dim: int):
        self._dim = dim

    def _build_component(self) -> DummyJaxComponent:
        return DummyJaxComponent(
            dim=self._dim,
            model=self.model,
            gain=self.gain,
            gain_function=self.gain_fn,
            mask_idxs=self.mask_idxs,
        )


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

    def __set_gain_and_check(self, gain: ArrayOrFloat):
        """Checking that gain is transformed into jnp.ndarray, and values are correct"""
        self.component.update_gain(gain)
        np.testing.assert_array_equal(self.component.gain, gain)

    def test_update_gain(self):
        """Testing setting up different gains dimensions"""
        # Test scalar gain assignment
        self.__set_gain_and_check(1.0)
        self.__set_gain_and_check(np.array(2.0))
        self.__set_gain_and_check(jnp.array(3.0))

        # Test vector gain assignment
        self.__set_gain_and_check(np.zeros(self.dummy_dim))
        self.__set_gain_and_check(jnp.ones(self.dummy_dim))

        # Test assigning gain with other dimension
        self.set_dim()
        gain = np.eye(self.component.dim + 1)
        with self.assertRaises(ValueError):
            self.component.update_gain(gain)

    def test_vector_gain(self):
        """Testing proper convertation of gain to vector form"""
        # Vector gain could not be constucted till dimension is specified
        self.component.update_gain(1.0)
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


unittest.main()
