:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/configuration.rst

=============
Configuration
=============

The configuration module serves as the mathematical foundation of MJINX, providing essential utilities for robot kinematics, transformations, and state management. These core functions enable precise control and manipulation of robotic systems throughout the library.

The module is structured into three complementary categories, each addressing a critical aspect of robot configuration:

1. **Model** - Functions for manipulating the MuJoCo model and managing robot state
2. **Lie Algebra** - Specialized tools for handling rotations and transformations with mathematical rigor
3. **Collision** - Algorithms for detecting and responding to potential collisions

******
Model
******

The model operations provide fundamental capabilities for state integration, Jacobian computation, and frame transformations. These functions form the bridge between abstract mathematical representations and the physical robot model.


.. automodule:: mjinx.configuration._model
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

************
Lie Algebra
************

The Lie algebra operations implement sophisticated mathematical tools for handling rotations, quaternions, and transformations in 3D space. Based on principles from differential geometry, these functions ensure proper handling of the SE(3) and SO(3) Lie groups.

.. automodule:: mjinx.configuration._lie
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

**********
Collision
**********

The collision operations provide sophisticated algorithms for detecting potential collisions, computing distances between objects, and analyzing contact points. These functions are crucial for implementing safety constraints and realistic physical interactions.

.. automodule:: mjinx.configuration._collision
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource