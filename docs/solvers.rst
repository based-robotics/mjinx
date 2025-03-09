:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/solvers.rst

.. _Solvers:

========
Solvers
========

MJINX provides multiple solver implementations for inverse kinematics problems, each with different characteristics suitable for various applications.

***********
Base Solver
***********

The abstract base class defining the interface for all solvers.

.. automodule:: mjinx.solvers._base
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource


***************
Local IK Solver
***************

A Quadratic Programming (QP) based solver that linearizes the problem at each step. This solver is efficient for real-time control and tracking applications.


The Local IK Solver includes an analytical solver feature that can significantly accelerate computation for problems with only velocity limits (no other inequality constraints). When enabled (via the ``use_analytical_solver=True`` parameter), the solver will:

1. For problems with no equality constraints, directly compute the solution as :math:`v = -P^{-1}c`
2. For problems with equality constraints, solve the KKT system:

   .. math::

       \begin{bmatrix} P & E^T \\ E & 0 \end{bmatrix} \begin{bmatrix} v \\ \lambda \end{bmatrix} = \begin{bmatrix} -c \\ d \end{bmatrix}

3. In both cases, clip the solution to satisfy velocity limits: :math:`v_{clipped} = \text{clip}(v, v_{min}, v_{max})`

This approach is much faster than solving the full QP problem, often providing a 5-10x speedup while maintaining good solution quality. For problems with other inequality constraints (barriers), the solver automatically falls back to the OSQP solver.

.. automodule:: mjinx.solvers._local_ik
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

****************
Global IK Solver
****************

A nonlinear optimization solver that directly optimizes joint positions. This solver can find solutions that avoid local minima and is suitable for complex positioning tasks.

.. automodule:: mjinx.solvers._global_ik
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource