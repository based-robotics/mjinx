:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/index.rst

.. title:: Table of Contents

#####
MJINX
#####

.. raw:: html

   <div class="badge-container">
   
|colab| |pypi_version| |pypi_downloads|

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/based-robotics/mjinx/blob/main/examples/notebooks/turoial.ipynb
   :alt: Open in Colab

.. |pypi_version| image:: https://img.shields.io/pypi/v/mjinx?color=blue
   :target: https://pypi.org/project/mjinx/
   :alt: PyPI version

.. |pypi_downloads| image:: https://img.shields.io/pypi/dm/mjinx?color=blue
   :target: https://pypistats.org/packages/mjinx
   :alt: PyPI downloads

**MJINX** is a high-performance library for differentiable inverse kinematics, powered by `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ 
and `MuJoCo MJX <https://mujoco.readthedocs.io/en/stable/mjx.html>`_. The library was inspired by the Pinocchio-based tool `PINK <https://github.com/stephane-caron/pink/tree/main>`_ and Mujoco-based analogue `MINK <https://github.com/kevinzakka/mink/tree/main>`_.

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 15px;">
     <img src="https://github.com/based-robotics/mjinx/raw/main/img/local_ik_output.gif" style="width: 200px;" alt="KUKA arm example">
     <img src="https://github.com/based-robotics/mjinx/raw/main/img/go2_stance.gif" style="width: 200px;" alt="GO2 robot example">
     <img src="https://github.com/based-robotics/mjinx/raw/main/img/g1_heart.gif" style="width: 200px;" alt="Heart path example">
     <img src="https://github.com/based-robotics/mjinx/raw/main/img/cassie_caravan.gif" style="width: 200px;" alt="Heart path example">
   </div>

*************
Key Features
*************

1. **Flexibility**: Each control problem is assembled via ``Components``, which enforce desired behavior or keep the system within a safety set.
2. **Multiple solution approaches**: JAX's efficient sampling and autodifferentiation enable various solvers optimized for different scenarios.
3. **Fully JAX-compatible**: Both the optimization problem and solver support JAX transformations, including JIT compilation and automatic vectorization.
4. **Convenience**: The API is designed to make complex inverse kinematics problems easy to express and solve.

*************
Citing MJINX
*************

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


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    installation.rst
    quick_start.rst
    problem.rst
    configuration.rst
    components.rst
    solvers.rst
    visualization.rst
    references.rst
    developer-notes.rst


