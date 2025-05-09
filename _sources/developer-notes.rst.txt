:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/developer-notes.rst

=================
Developer Notes
=================

This section contains information for developers who want to contribute to MJINX or understand its internals better.

*******************
Code Organization
*******************

MJINX follows a modular architecture:

- ``mjinx/components/`` - Task and barrier implementations
- ``mjinx/configuration/`` - Kinematics and transformation utilities
- ``mjinx/solvers/`` - Inverse kinematics solvers
- ``mjinx/visualize.py`` - Visualization tools
- ``mjinx/problem.py`` - Problem construction and management
- ``mjinx/typing.py`` - Type definitions

**************************
Development Guidelines
**************************

When contributing to MJINX, please follow these guidelines:

1. **Type annotations**: Use type annotations throughout the code.
2. **Documentation**: Write clear docstrings with reStructuredText format.
3. **Testing**: Add tests for new features using pytest.
4. **JAX compatibility**: Ensure new code works with JAX transformations.
5. **Performance**: Consider computation efficiency, especially for operations in inner loops.

******************
JAX Considerations
******************

MJINX leverages JAX for automatic differentiation and acceleration. When working with JAX:

- Use JAX's functional programming style
- Avoid in-place mutations
- Remember that JAX arrays are immutable
- Use ``jit``, ``vmap``, and ``grad`` to leverage JAX's transformations
- Test code with both CPU and GPU devices

For more information, refer to the `JAX documentation <https://jax.readthedocs.io/>`_.

****************
Types
****************

MJINX uses a comprehensive type system to ensure code correctness and improve developer experience. Understanding the type system is essential for contributing to the codebase and extending its functionality.

The type system provides clear interfaces between components, helps catch errors at development time, and makes the code more maintainable. All new contributions should adhere to the established typing conventions.

For detailed information about MJINX's type system, including type aliases and enumerations, see the :doc:`typing` documentation.

.. toctree::
   :maxdepth: 2

   typing
