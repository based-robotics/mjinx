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