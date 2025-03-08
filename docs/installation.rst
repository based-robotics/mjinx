:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/installation.rst

************
Installation
************

===========
From PyPI
===========

The simplest way to install MJINX is via the Python Package Index:

.. code:: bash

    pip install mjinx

====================
Installation Options
====================

MJINX offers several installation options with different dependencies:

1. **Visualization tools**: ``pip install mjinx[visual]``
2. **Example dependencies**: ``pip install mjinx[examples]``
3. **Development dependencies**: ``pip install mjinx[dev]`` (preferably in editable mode)
4. **Documentation dependencies**: ``pip install mjinx[docs]``
5. **Complete installation**: ``pip install mjinx[all]``

============================
From Source (Developer Mode)
============================

For developers or to access the latest features, you can clone the repository and install in editable mode:

.. code:: bash

    git clone https://github.com/based-robotics/mjinx.git
    cd mjinx
    pip install -e .

With editable mode, any changes you make to the source code take effect immediately without requiring reinstallation.

For development work, we recommend installing with the development extras:

.. code:: bash

    pip install -e ".[dev]"