:github_url: https://github.com/based-robotics/mjinx/tree/main/docs/components.rst

==========
Components
==========

MJINX employs a component-based architecture to formulate inverse kinematics problems through functional decomposition. Each component encapsulates a mathematical mapping from the configuration space to a task or constraint space, functioning either as an objective function (task) or an inequality constraint (barrier).

This modular structure facilitates the systematic construction of complex kinematic problems through composition of elementary components. The approach aligns with established practices in robotics control theory, where complex behaviors emerge from the coordination of simpler control objectives.

Components are characterized by several key attributes:

- A unique identifier for reference within the problem formulation
- A gain parameter that determines its relative weight in the optimization
- An optional dimensional mask for selective application
- A differentiable function mapping robot state to output values

This formulation follows the task-priority framework established in robotics literature, where multiple objectives are managed through appropriate weighting and prioritization. The separation of concerns between tasks and constraints provides a natural expression of both the desired behavior and the feasible region of operation.

When integrated into a Problem instance, components form a well-posed optimization problem. Tasks define the objective function to be minimized, while barriers establish the constraint manifold. The solver then computes solutions that optimize the weighted task objectives while maintaining feasibility with respect to all constraints.

**************
Base Component
**************

The Component class serves as the abstract base class from which all specific component implementations derive. This inheritance hierarchy ensures a consistent interface while enabling specialized behavior for different component types.

.. automodule:: mjinx.components._base
    :members:
    :special-members: __call__
    :member-order: bysource

********
Tasks
********

Tasks define objective functions that map from configuration space to task space, with the solver minimizing weighted errors between current and desired values :cite:`kanoun2011kinematic`. Each task :math:`f: \mathcal{Q} \rightarrow \mathbb{R}^m` produces an error :math:`e(q) = f(q) - f_{desired}` that is minimized according to :math:`\|e(q)\|^2_W`.

MJINX provides task implementations for common robotics objectives:

.. toctree::
   :maxdepth: 1

   tasks

********
Barriers
********
Barriers implement inequality constraints through scalar functions :math:`h(q) \geq 0` that create boundaries the solver must respect. Based on control barrier functions (CBFs) :cite:`ames2019control`, these constraints are enforced through differential inequality: :math:`\nabla h(q)^T v \geq -\alpha h(q)`, with :math:`\alpha` controls constraint enforcement and :math:`v` is the velocity vector. 

MJINX provides several barrier implementations:

.. toctree::
   :maxdepth: 1

   barriers

***********
Constraints
***********

Constraints represent a new type of component with a strictly enforced equality condition :math:`f(q) = 0`. Those constraints might be either treated strictly as differentiated exponentially stable equality: :math:`\nabla h(q)^T v = -\alpha h(q)`, with :math:`\alpha` controls constraint enforcement and :math:`v` is the velocity vector, or as a soft constraint -- task with high gain.

Yet, only the following constraints are implemented:

.. toctree::
   :maxdepth: 1

   constraints