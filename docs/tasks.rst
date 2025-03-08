:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/tasks.rst

.. _Tasks:

Tasks
=====

Tasks define the objectives or goals in an inverse kinematics problem. Each task represents a function that the solver attempts to minimize, driving the robot toward desired configurations or behaviors.

MJINX provides task implementations for common robotics objectives:
- Controlling specific joints
- Positioning robot end-effectors
- Setting object orientations
- Managing the center of mass
- And more

All tasks inherit from the base Task class and follow a common interface, making them easily interchangeable and composable.

Base Task
---------
The foundational class that all tasks extend. It defines the core interface and functionality for task objectives.

.. automodule:: mjinx.components.tasks._base
    :members:

Center of Mass Task
-------------------
Controls the position of the robot's center of mass, useful for balance and stability.

.. automodule:: mjinx.components.tasks._com_task
    :members:

Joint Task
----------
Directly controls joint positions, useful for regularization and posture optimization.

.. automodule:: mjinx.components.tasks._joint_task
    :members:

Base Body Task
--------------
The foundation for tasks that target specific bodies, geometries, or sites in the robot model.

.. automodule:: mjinx.components.tasks._obj_task
    :members:

Body Frame Task
---------------
Controls the complete pose (position and orientation) of a body, geometry, or site.

.. automodule:: mjinx.components.tasks._obj_frame_task
    :members:

Body Position Task
------------------
Controls just the position of a body, geometry, or site, ignoring orientation.

.. automodule:: mjinx.components.tasks._obj_position_task
    :members:
