# import unittest

# import jax
# import jax.numpy as jnp
# import jaxlie
# import mujoco as mj
# from jaxlie import SE3, SO3
# from mujoco import mjx

# # Import the functions to be tested
# from mjinx.configuration import (
#     check_limits,
#     get_configuration_limit,
#     get_frame_jacobian_local,
#     get_frame_jacobian_world_aligned,
#     get_joint_zero,
#     get_transform,
#     get_transform_frame_to_world,
#     integrate,
#     joint_difference,
#     update,
# )


# class TestConfiguration(unittest.TestCase):
#     def setUp(self):
#         # Create a simple MuJoCo model for testing
#         self.model = mjx.Model.from_xml(
#             """
#         <mujoco>
#           <worldbody>
#             <body name="body1" pos="0 0 0">
#               <joint name="joint1" type="hinge"/>
#               <geom size="1"/>
#             </body>
#           </worldbody>
#         </mujoco>
#         """
#         )
#         self.q = jnp.array([0.5])  # Example joint position

#     def test_update(self):
#         data = update(self.model, self.q)
#         self.assertIsInstance(data, mjx.Data)
#         self.assertTrue(jnp.allclose(data.qpos, self.q))

#     def test_check_limits(self):
#         data = update(self.model, self.q)
#         result = check_limits(self.model, data)
#         self.assertIsInstance(result, bool)

#     def test_get_frame_jacobian_world_aligned(self):
#         data = update(self.model, self.q)
#         jacobian = get_frame_jacobian_world_aligned(self.model, data, 1)
#         self.assertEqual(jacobian.shape, (self.model.nv, 6))

#     def test_get_frame_jacobian_local(self):
#         data = update(self.model, self.q)
#         jacobian = get_frame_jacobian_local(self.model, data, 1)
#         self.assertEqual(jacobian.shape, (self.model.nv, 6))

#     def test_get_transform_frame_to_world(self):
#         data = update(self.model, self.q)
#         transform = get_transform_frame_to_world(self.model, data, 1)
#         self.assertIsInstance(transform, SE3)

#     def test_get_transform(self):
#         data = update(self.model, self.q)
#         transform = get_transform(self.model, data, 0, 1)
#         self.assertIsInstance(transform, SE3)

#     def test_integrate(self):
#         velocity = jnp.array([0.1])
#         dt = jnp.array(0.1)
#         q_new = integrate(self.model, self.q, velocity, dt)
#         self.assertEqual(q_new.shape, self.q.shape)

#     def test_get_configuration_limit(self):
#         limit = 1.0
#         A, b = get_configuration_limit(self.model, limit)
#         self.assertEqual(A.shape, (2 * self.model.nv, self.model.nv))
#         self.assertEqual(b.shape, (2 * self.model.nv,))

#     def test_get_joint_zero(self):
#         zero_config = get_joint_zero(self.model)
#         self.assertEqual(zero_config.shape, (self.model.nq,))

#     def test_joint_difference(self):
#         q1 = jnp.array([0.5])
#         q2 = jnp.array([0.7])
#         diff = joint_difference(self.model, q1, q2)
#         self.assertEqual(diff.shape, (self.model.nv,))

#     # Additional tests for edge cases and different joint types
#     def test_get_joint_zero_complex_model(self):
#         complex_model = mjx.Model.from_xml(
#             """
#         <mujoco>
#           <worldbody>
#             <body name="body1" pos="0 0 0">
#               <joint name="free_joint" type="free"/>
#               <body name="body2">
#                 <joint name="ball_joint" type="ball"/>
#                 <body name="body3">
#                   <joint name="hinge_joint" type="hinge"/>
#                   <body name="body4">
#                     <joint name="slide_joint" type="slide"/>
#                   </body>
#                 </body>
#               </body>
#             </body>
#           </worldbody>
#         </mujoco>
#         """
#         )
#         zero_config = get_joint_zero(complex_model)
#         expected_length = 7 + 4 + 1 + 1  # free + ball + hinge + slide
#         self.assertEqual(zero_config.shape, (expected_length,))

#     def test_joint_difference_complex_model(self):
#         complex_model = mjx.Model.from_xml(
#             """
#         <mujoco>
#           <worldbody>
#             <body name="body1" pos="0 0 0">
#               <joint name="free_joint" type="free"/>
#               <body name="body2">
#                 <joint name="ball_joint" type="ball"/>
#                 <body name="body3">
#                   <joint name="hinge_joint" type="hinge"/>
#                   <body name="body4">
#                     <joint name="slide_joint" type="slide"/>
#                   </body>
#                 </body>
#               </body>
#             </body>
#           </worldbody>
#         </mujoco>
#         """
#         )
#         q1 = jnp.array(
#             [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.5, 0.2]  # free joint  # ball joint  # hinge joint
#         )  # slide joint
#         q2 = jnp.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0.7, 0.3])
#         diff = joint_difference(complex_model, q1, q2)
#         expected_length = 6 + 3 + 1 + 1  # free + ball + hinge + slide
#         self.assertEqual(diff.shape, (expected_length,))

#     def test_get_configuration_limit_array(self):
#         limit_array = jnp.array([0.5, 1.0, 1.5])
#         model = mjx.Model.from_xml(
#             """
#         <mujoco>
#           <worldbody>
#             <body name="body1">
#               <joint name="joint1" type="slide"/>
#             </body>
#             <body name="body2">
#               <joint name="joint2" type="hinge"/>
#             </body>
#             <body name="body3">
#               <joint name="joint3" type="ball"/>
#             </body>
#           </worldbody>
#         </mujoco>
#         """
#         )
#         A, b = get_configuration_limit(model, limit_array)
#         self.assertEqual(A.shape, (6, 3))
#         self.assertTrue(jnp.allclose(b, jnp.array([0.5, 1.0, 1.5, 0.5, 1.0, 1.5])))


# if __name__ == "__main__":
#
