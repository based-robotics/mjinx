:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/typing.rst

.. _typing:

Typing
============

.. automodule:: mjinx.typing
   :members:
   :undoc-members:
   :show-inheritance:

Type Aliases
------------

.. data:: ndarray
   :no-index:

   Type alias for numpy or JAX numpy arrays.

   :annotation: = np.ndarray | jnp.ndarray

.. data:: ArrayOrFloat
   :no-index:

   Type alias for an array or a float value.

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

Enumerations
------------

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