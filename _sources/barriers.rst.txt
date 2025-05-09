:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/barriers.rst

.. _Barriers:


Barriers implement inequality constraints in the inverse kinematics problem. Each barrier defines a scalar function :math:`h: \mathcal{Q} \rightarrow \mathbb{R}` that must remain positive (:math:`h(q) \geq 0`), creating a boundary that the solver must respect. This approach is mathematically equivalent to control barrier functions (CBFs) :cite:`ames2019control`, which have been widely adopted in robotics for safety-critical control.

The barrier formulation creates a continuous constraint boundary that prevents the system from entering prohibited regions of the configuration space. In the optimization problem, these constraints are typically enforced through linearization at each step:

.. math::

    \nabla h(q)^T \dot{q} \geq -\alpha h(q)

where :math:`\alpha` is a gain parameter that controls how aggressively the system is pushed away from the constraint boundary.


All barriers follow a consistent mathematical formulation while adapting to specific constraint types, enabling systematic enforcement of safety and feasibility requirements.

Base Barrier
------------
The foundation for all barrier constraints, defining the core mathematical properties and interface.

.. automodule:: mjinx.components.barriers._base
    :members:
    :member-order: bysource

Joint Barrier
-------------
Enforces joint limit constraints, preventing the robot from exceeding mechanical limits.

.. automodule:: mjinx.components.barriers._joint_barrier
    :members:
    :member-order: bysource

Base Body Barrier
-----------------
The foundation for barriers applied to specific bodies, geometries, or sites in the robot model. This abstract class provides common functionality for all object-specific barriers.

.. automodule:: mjinx.components.barriers._obj_barrier
    :members:
    :member-order: bysource

Body Position Barrier
---------------------
Enforces position constraints on specific bodies, geometries, or sites, useful for defining workspace limits.

.. automodule:: mjinx.components.barriers._obj_position_barrier
    :members:
    :member-order: bysource

Self Collision Barrier
----------------------
Prevents different parts of the robot from colliding with each other, essential for complex manipulators.

.. automodule:: mjinx.components.barriers._self_collision_barrier
    :members:
    :member-order: bysource