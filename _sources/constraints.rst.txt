:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/constraints.rst

.. _Constraints:

Constraints represent a new type of component with a strictly enforced equality, either hardly or softly.

The constratin is formulated as :math:`f(q) = 0`. Those constraints might be either treated strictly as differentiated exponentially stable equality: :math:`\nabla h(q)^T v = -\alpha h(q)`, with :math:`\alpha` controls constraint enforcement and :math:`v` is the velocity vector, or as a soft constraint -- task with high gain.

Base Constraint
---------------
The foundational class that all constraints extend. It defines the core interface and mathematical properties for constraint objectives.

.. automodule:: mjinx.components.constraints._base
    :members:
    :member-order: bysource

Model Equality Constraint
-------------------------
The constraints, described in MuJoCo model as equality constrainst. 

.. automodule:: mjinx.components.constraints._equality_constraint
    :members:
    :member-order: bysource

.. Base Task
.. ---------
.. The foundational class that all tasks extend. It defines the core interface and mathematical properties for task objectives.

.. .. automodule:: mjinx.components.tasks._base
..     :members:
..     :member-order: bysource
