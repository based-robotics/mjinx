:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/problem.rst

.. _Problem:

=======
Problem
=======

At the heart of MJINX lies the Problem module - a structured framework that elegantly handles inverse kinematics challenges. This module serves as the central hub where various components come together to form a cohesive mathematical formulation.

When working with MJINX, a Problem instance orchestrates several key elements:

- **Tasks**: Objective functions that define desired behaviors, such as reaching specific poses or following trajectories
- **Barriers**: Smooth constraint functions that naturally keep the system within valid states
- **Velocity limits**: Physical bounds on joint velocities to ensure feasible motion

The module's modular architecture shines in its flexibility - users can begin with simple scenarios like positioning a single end-effector, then naturally build up to complex whole-body motions with multiple objectives and constraints.

Under the hood, the Problem class transforms these high-level specifications into optimized computations through its JaxProblemData representation. By leveraging JAX's JIT compilation, it ensures that even sophisticated inverse kinematics problems run with maximum efficiency.

.. automodule:: mjinx.problem
    :members:
    :member-order: bysource

