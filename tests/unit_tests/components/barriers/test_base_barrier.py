import unittest

import jax.numpy as jnp
import mujoco.mjx as mjx

from mjinx.components.barriers import Barrier, JaxBarrier


class DummyJaxBarrier(JaxBarrier):
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        return jnp.ones(self.model.nv)[self.mask_idxs,]


class DummyBarrier(Barrier[DummyJaxBarrier]):
    JaxComponentType: type = DummyJaxBarrier


class TestBarrier(unittest.TestCase):
    def test_safe_displacement_gain(self):
        self.component = DummyBarrier("test_component", gain=2.0)
        self.component.safe_displacement_gain = 5.0
        self.assertEqual(self.component.safe_displacement_gain, 5.0)
