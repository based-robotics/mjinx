:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/typing.rst

************
Type System
************

MJINX uses Python's type annotations throughout the codebase to enhance code clarity, enable better IDE support, and catch potential errors. This module provides the type definitions and aliases used across the library.

.. automodule:: mjinx.typing
   :members:
   :undoc-members:
   :show-inheritance:

************
Type Aliases
************

The following type aliases are defined for common data structures and function signatures:

.. data:: ndarray
   :no-index:

   Type alias for numpy or JAX numpy arrays.

   :annotation: = np.ndarray | jnp.ndarray

.. data:: ArrayOrFloat
   :no-index:

   Type alias for an array or a scalar float value.

   :annotation: = ndarray | float

.. data:: ClassKFunctions
   :no-index:

   Type alias for Class K functions, which are scalar functions that take and return ndarrays.

   :annotation: = Callable[[ndarray], ndarray]

.. data:: CollisionBody
   :no-index:

   Type alias for collision body representation, either as an integer ID or a string name.

   :annotation: = int | str

.. data:: CollisionPair
   :no-index:

   Type alias for a pair of collision body IDs.

   :annotation: = tuple[int, int]

************
Enumerations
************

.. autoclass:: PositionLimitType
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   Enumeration of possible position limit types.

   .. attribute:: MIN
      :no-index:
      :value: 0

      Minimum position limit.

   .. attribute:: MAX
      :no-index:
      :value: 1

      Maximum position limit.

   .. attribute:: BOTH
      :no-index:
      :value: 2

      Both minimum and maximum position limits.