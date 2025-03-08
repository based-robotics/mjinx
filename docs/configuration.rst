:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/configuration.rst

.. _Configuration:

Configuration
=============

The configuration module provides essential utilities for working with robot kinematics, transformations, and state management. These functions form the mathematical foundation of MJINX and are used extensively throughout the library.

The module is organized into three main categories:
1. Model operations - Functions for working with the MuJoCo model and robot state
2. Lie algebra operations - Functions for properly handling rotations and transformations
3. Collision operations - Functions for detecting and managing object collisions

Model
-----
Functions for state integration, Jacobian computation, and frame transformations.

.. automodule:: mjinx.configuration._model
    :members:
    :undoc-members:
    :show-inheritance:

Lie Algebra
-----------
Functions for handling rotations, quaternions, and mathematical operations on the SE(3) and SO(3) Lie groups.

.. automodule:: mjinx.configuration._lie
    :members:
    :undoc-members:
    :show-inheritance:

Collisions
----------
Functions for collision detection, distance computation, and contact point analysis.

.. automodule:: mjinx.configuration._collision
    :members:
    :undoc-members:
    :show-inheritance: