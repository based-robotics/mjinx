:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/barriers.rst

.. _Barriers:

Barriers
========

Barriers represent constraints in the inverse kinematics problem. Each barrier defines a mathematical function that must remain positive (h(q) ≥ 0), creating a "barrier" that the solver must not cross.

MJINX provides barrier implementations for common robotics constraints:
- Joint limits
- Position boundaries
- Collision avoidance
- And more

All barriers follow a consistent mathematical formulation while adapting to specific constraint types.

Base Barrier
------------
The foundation for all barrier constraints, defining the core interface and mathematical properties.

.. automodule:: mjinx.components.barriers._base
    :members:

Joint Barrier
-------------
Enforces joint limit constraints, preventing the robot from exceeding mechanical limits.

.. automodule:: mjinx.components.barriers._joint_barrier
    :members:

Base Body Barrier
-----------------
The foundation for barriers applied to specific bodies, geometries, or sites in the robot model.

.. automodule:: mjinx.components.barriers._obj_barrier
    :members:

Body Position Barrier
---------------------
Enforces position constraints on specific bodies, geometries, or sites, useful for workspace limits.

.. automodule:: mjinx.components.barriers._obj_position_barrier
    :members:

Self Collision Barrier
----------------------
Prevents different parts of the robot from colliding with each other, essential for complex manipulators.

.. automodule:: mjinx.components.barriers._self_collision_barrier
    :members: