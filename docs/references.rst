:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/references.rst

**********
References
**********

=====================
Academic References
=====================

This section lists key academic papers and resources that influenced MJINX's design and implementation:

Citations
---------

This project builds on the following academic works:

.. bibliography::
   :style: plain
   :all:

===================
Software References
===================

MJINX builds upon and integrates with the following software libraries:

- `JAX <https://github.com/google/jax>`_: Autograd and XLA for high-performance machine learning research.
- `MuJoCo <https://mujoco.org/>`_: Physics engine for detailed, efficient robot simulation.
- `PINK <https://github.com/stephane-caron/pink>`_: Differentiable inverse kinematics using Pinocchio.
- `MINK <https://github.com/kevinzakka/mink>`_: MuJoCo-based inverse kinematics.
- `Optax <https://github.com/deepmind/optax>`_: Gradient processing and optimization.
- `JaxLie <https://github.com/brentyi/jaxlie>`_: JAX library for Lie groups.

=================
Acknowledgements
=================

MJINX would not exist without the contributions and inspiration from several sources:

- Simeon Nedelchev for guidance and contributions during development
- Stéphane Caron and Kevin Zakka, whose work on PINK and MINK respectively provided significant inspiration
- The MuJoCo MJX team for their excellent physics simulation tools
- IRIS lab at KAIST


=============
Citing MJINX
=============

If you use MJINX in your research, please cite it as follows:

.. code-block:: bibtex

    @software{mjinx25,
    author = {Domrachev, Ivan and Nedelchev, Simeon},
    license = {MIT},
    month = mar,
    title = {{MJINX: Differentiable GPU-accelerated inverse kinematics in JAX}},
    url = {https://github.com/based-robotics/mjinx},
    version = {0.1.1},
    year = {2025}
    }


