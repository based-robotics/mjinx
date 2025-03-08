:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/tasks.rst

.. _Tasks:

Tasks define the objective functions in an inverse kinematics problem. Each task represents a mapping from the configuration space to a task space, with the solver minimizing the weighted error between the current and desired task values. This approach follows the task-priority framework established in robotics literature :cite:`kanoun2011kinematic`, where multiple objectives are managed through appropriate weighting and prioritization.

Mathematically, a task is defined as a function :math:`f: \mathcal{Q} \rightarrow \mathbb{R}^m` that maps from the configuration space :math:`\mathcal{Q}` to a task space. The error is computed as :math:`e(q) = f(q) - f_{desired}`, and the solver minimizes a weighted norm of this error: :math:`\|e(q)\|^2_W`, where :math:`W` is a positive-definite weight matrix.

All tasks inherit from the base Task class and follow a consistent mathematical formulation, enabling systematic composition of complex behaviors through the combination of elementary tasks.

Base Task
---------
The foundational class that all tasks extend. It defines the core interface and mathematical properties for task objectives.

.. automodule:: mjinx.components.tasks._base
    :members:

Center of Mass Task
-------------------
Controls the position of the robot's center of mass, critical for maintaining balance in legged systems and manipulators.

.. automodule:: mjinx.components.tasks._com_task
    :members:

Joint Task
----------
Directly controls joint positions, useful for regularization, posture optimization, and redundancy resolution.

.. automodule:: mjinx.components.tasks._joint_task
    :members:

Base Body Task
--------------
The foundation for tasks that target specific bodies, geometries, or sites in the robot model. This abstract class provides common functionality for all object-specific tasks.

.. automodule:: mjinx.components.tasks._obj_task
    :members:

Body Frame Task
---------------
Controls the complete pose (position and orientation) of a body, geometry, or site using SE(3) representations.

.. automodule:: mjinx.components.tasks._obj_frame_task
    :members:

Body Position Task
------------------
Controls just the position of a body, geometry, or site, ignoring orientation. Useful when only positional constraints matter.

.. automodule:: mjinx.components.tasks._obj_position_task
    :members:
