:github_url: https://github.com/based-robotics/mjinx/tree/main/docs/components.rst

.. _Components:

Components
==========

Components are the building blocks of MJINX problems. Each component represents a mathematical function that contributes to the inverse kinematics problem, either as a task (objective) or a barrier (constraint).

Components have several key properties:
- A unique name for identification
- A gain (weight) that determines its importance in the optimization
- An optional mask that selects relevant dimensions
- A mathematical function that maps robot state to output values

The component system in MJINX is designed to be modular and composable, allowing you to build complex problems from simple elements.

.. toctree::
   :maxdepth: 2
   :caption: Component Types:

   tasks
   barriers

Base Component
--------------
All components inherit from the base Component class, which provides the core interface and functionality.

.. automodule:: mjinx.components._base
    :members:
    :special-members: __call__
