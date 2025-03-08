:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/problem.rst

.. _Problem:

Problem
=======

The Problem module is central to MJINX, providing the framework for defining and compiling inverse kinematics problems.

A Problem consists of:
- Tasks: Components representing desired behaviors to achieve
- Barriers: Components representing constraints to satisfy
- Velocity limits: Bounds on allowable joint velocities

Once defined, a Problem is compiled into a JaxProblemData object that can be passed to solvers.

.. automodule:: mjinx.problem
    :members:

