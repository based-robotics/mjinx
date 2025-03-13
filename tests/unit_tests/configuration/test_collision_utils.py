import math
import unittest

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np
from mujoco import mjx

# For monkey-patching the collision function
from mjinx.configuration import _collision  # to refer to _COLLISION_FUNC
from mjinx.configuration._collision import (
    compute_collision_pairs,
    geom_groups,
    get_distance,
    sorted_pair,
)

COLLISION_FUNC = _collision.__dict__.get("_COLLISION_FUNC")


class DummyModel:
    """A simple dummy model to mimic necessary fields for testing geom_groups."""

    def __init__(self):
        # For a model with, say, 2 geometries.
        self.geom_type = np.array([None, None])
        self.geom_dataid = np.array([None, None])
        self.geom_priority = np.array([None, None])
        self.geom_condim = np.array([None, None])
        # For hfield branch (only needed when a geom is a height field)
        self.hfield_nrow = None
        self.hfield_ncol = None
        self.hfield_size = None
        self.geom_rbound = np.array([])


class TestCollisionExtended(unittest.TestCase):
    def test_sorted_pair(self):
        """Test sorted_pair."""
        self.assertEqual(sorted_pair(5, 3), (3, 5))
        self.assertEqual(sorted_pair(2, 2), (2, 2))

    def test_geom_groups_ordering(self):
        """
        Test that geom_groups swaps geometry indices when the first geom's type is greater.
        """
        dummy = DummyModel()
        # Assign dummy values:
        # Let geom 0 have a higher type than geom 1 so that they get swapped.
        dummy.geom_type = np.array([5, 3])
        dummy.geom_dataid = np.array([10, 11])
        # Set same priority so branch falls in the "else" (max of condim)
        dummy.geom_priority = np.array([0, 0])
        dummy.geom_condim = np.array([7, 5])
        # Provide a single collision pair (using unsorted order)
        collision_pairs = [(0, 1)]
        groups = geom_groups(dummy, collision_pairs)
        # There should be one group with a key that uses the sorted types.
        key = list(groups.keys())[0]
        # After swapping, types should be (3,5)
        self.assertEqual(key.types, (3, 5))
        # Since priorities are equal, condim is max(7,5) = 7.
        self.assertEqual(key.condim, 7)

    def test_geom_groups_priority(self):
        """
        Test that geom_groups selects condim based on priority.
        """
        dummy = DummyModel()
        dummy.geom_type = np.array([int(mj.mjtGeom.mjGEOM_BOX), int(mj.mjtGeom.mjGEOM_BOX)])
        dummy.geom_dataid = np.array([0, 1])
        # Set different priorities: first geom has higher priority.
        dummy.geom_priority = np.array([2, 1])
        dummy.geom_condim = np.array([4, 3])
        collision_pairs = [(0, 1)]
        groups = geom_groups(dummy, collision_pairs)
        key = list(groups.keys())[0]
        # Since geom_priority[0] > geom_priority[1], condim should be that of geom 0.
        self.assertEqual(key.condim, 4)

    def test_compute_collision_pairs_empty(self):
        """
        Test that compute_collision_pairs returns an empty contact when collision_pairs is empty.
        """
        # Create a simple model from XML (using a sphere-sphere example)
        xml = """
        <mujoco>
          <worldbody>
            <body name="a">
              <geom name="geom_a" type="sphere" size="0.5"/>
            </body>
            <body name="b">
              <geom name="geom_b" type="sphere" size="0.5"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mjx.put_model(mj.MjModel.from_xml_string(xml))
        data = mjx.make_data(model)
        contact = compute_collision_pairs(model, data, [])
        # Expect empty arrays
        self.assertEqual(contact.dist.shape[0], 0)
        self.assertEqual(contact.pos.size, 0)
        self.assertEqual(contact.frame.size, 0)

    def test_get_distance_normal(self):
        """
        Test get_distance on non-hfield geoms.
        Monkey-patch _COLLISION_FUNC for sphere-sphere collisions.
        """

        # Define a dummy collision function that returns fixed arrays.
        def dummy_collision_fn(model, data, key, geom_array):
            # Return dist with two candidate values (min should be 3.0),
            # a contact position, and a frame.
            return jnp.array([[3.0, 4.0]]), jnp.array([[0.0, 0.0, 0.0]]), jnp.array([[0.0, 0.0, 1.0]])

        # Determine key for two spheres.
        sphere_types = (mj.mjtGeom.mjGEOM_SPHERE, mj.mjtGeom.mjGEOM_SPHERE)
        # Save the current function, if any.
        orig_fn = COLLISION_FUNC.get(sphere_types, None)
        COLLISION_FUNC[sphere_types] = dummy_collision_fn

        xml = """
        <mujoco>
          <worldbody>
            <body name="a">
              <geom name="geom_a" type="sphere" size="0.5"/>
            </body>
            <body name="b">
              <geom name="geom_b" type="sphere" size="0.5"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mjx.put_model(mj.MjModel.from_xml_string(xml))
        data = mjx.make_data(model)
        collision_pairs = [(0, 1)]
        dists, pos, frame = get_distance(model, data, collision_pairs)
        # dummy returns [[3.0,4.0]] so dist.min() should be 3.0.
        self.assertTrue(jnp.allclose(dists, jnp.array([3.0])))
        self.assertEqual(pos.shape, (1, 3))
        self.assertEqual(frame.shape, (1, 3))
        # Restore original collision function if it existed.
        if orig_fn is not None:
            COLLISION_FUNC[sphere_types] = orig_fn
        else:
            del COLLISION_FUNC[sphere_types]

    def test_get_distance_hfield_raises(self):
        """
        Test that get_distance raises NotImplementedError when the first geom is an hfield.
        """
        xml = """
        <mujoco>
          <asset>
              <hfield name="hf" nrow="10" ncol="10" size="1 1 0.1 0.1"/>
          </asset>
          <worldbody>
            <geom name="hfield_geom" type="hfield" hfield="hf" pos="0 0 0" size="1 1 0.1"/>
            <body name="b">
              <geom name="geom_b" type="sphere" size="0.5"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mjx.put_model(mj.MjModel.from_xml_string(xml))
        data = mjx.make_data(model)
        # Set minimal attributes to avoid attribute errors.
        # In real cases these are set by MuJoCo.
        model = model.replace(
            geom_type=np.array([int(mj.mjtGeom.mjGEOM_HFIELD), int(mj.mjtGeom.mjGEOM_SPHERE)]),
            geom_dataid=np.array([0, 1]),
            geom_priority=np.array([0, 0]),
            geom_condim=np.array([3, 3]),
        )
        with self.assertRaises(NotImplementedError):
            get_distance(model, data, [(0, 1)])
