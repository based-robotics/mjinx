:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/references.rst

**********
References
**********

Academic References
==================

This section lists key academic papers and resources that influenced MJINX's design and implementation:

1. Stephane Caron. (2022). "Inverse Kinematics with Inequality Constraints: The PINK Library." IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

2. Del Prete, A. (2018). "Joint Position and Velocity Bounds in Discrete-Time Acceleration/Torque Control of Robot Manipulators." IEEE Robotics and Automation Letters, 3(1), 281-288.

3. Kanoun, O., Lamiraux, F., & Wieber, P. B. (2011). "Kinematic Control of Redundant Manipulators: Generalizing the Task-Priority Framework to Inequality Task." IEEE Transactions on Robotics, 27(4), 785-792.

4. Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., ... & Wanderman-Milne, S. (2018). "JAX: composable transformations of Python+NumPy programs."

Software References
=================

MJINX builds upon and integrates with the following software libraries:

- `JAX <https://github.com/google/jax>`_: Autograd and XLA for high-performance machine learning research.
- `MuJoCo <https://mujoco.org/>`_: Physics engine for detailed, efficient robot simulation.
- `PINK <https://github.com/stephane-caron/pink>`_: Differentiable inverse kinematics using Pinocchio.
- `MINK <https://github.com/kevinzakka/mink>`_: MuJoCo-based inverse kinematics.
- `Optax <https://github.com/deepmind/optax>`_: Gradient processing and optimization.
- `JaxLie <https://github.com/brentyi/jaxlie>`_: JAX library for Lie groups.

Acknowledgements
===============

MJINX would not exist without the contributions and inspiration from several sources:

- Simeon Nedelchev for guidance and contributions during development
- Stéphane Caron and Kevin Zakka, whose work on PINK and MINK respectively provided significant inspiration
- The MuJoCo MJX team for their excellent physics simulation tools
- IRIS lab at KAIST
