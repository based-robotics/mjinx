:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/quick_start.rst

============
Quick Start
============

Inverse kinematics (IK) is the computational problem of determining joint configurations that achieve desired end-effector poses or other system states. MJINX provides a structured framework for formulating and solving these problems through a component-based architecture that elegantly handles both objectives and constraints.

You can follow along with this guide interactively by clicking on |colab|. This notebook environment allows you to experiment with MJINX without local installation.

   .. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/based-robotics/mjinx/blob/main/examples/notebooks/turoial.ipynb
      :alt: Open in Colab

****************
Complete Example
****************

Here's a comprehensive example demonstrating MJINX's capabilities with a 7-DoF Kuka iiwa manipulator:

.. code-block:: python

    import jax
    import numpy as np
    from mujoco import mjx
    from mjinx.problem import Problem
    from mjinx.components.tasks import FrameTask
    from mjinx.components.barriers import JointBarrier
    from mjinx.solvers import LocalIKSolver
    from mjinx.configuration import integrate

    # Initialize the robot model using MuJoCo
    from robot_descriptions.iiwa14_mj_description import MJCF_PATH
    import mujoco as mj
    mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
    mjx_model = mjx.put_model(mj_model)

    # Create instance of the problem
    problem = Problem(mjx_model)

    # Add tasks to track desired behavior
    frame_task = FrameTask("ee_task", cost=1, gain=20, body_name="link7")
    problem.add_component(frame_task)

    # Add barriers to enforce constraints
    joints_barrier = JointBarrier("jnt_range", gain=10)
    problem.add_component(joints_barrier)

    # Initialize the solver
    solver = LocalIKSolver(mjx_model)
    
    # Initial configuration
    q = np.zeros(mjx_model.nq)
    dt = 1e-2

    # Initialize solver data
    solver_data = solver.init(q)

    # JIT-compile solve and integrate functions
    solve_jit = jax.jit(solver.solve)
    integrate_jit = jax.jit(integrate, static_argnames=["dt"])

    # Control loop
    for t in np.arange(0, 5, dt):
        # Update target and compile problem
        frame_task.target_frame = np.array([0.1 * np.sin(t), 0.1 * np.cos(t), 0.1, 1, 0, 0, 0])
        problem_data = problem.compile()

        # Solve the IK problem
        opt_solution, solver_data = solve_jit(q, solver_data, problem_data)

        # Integrate to get new configuration
        q = integrate_jit(
            mjx_model,
            q,
            opt_solution.v_opt,
            dt,
        )

Let's examine the key elements of this example to understand MJINX's approach to inverse kinematics.

**********************
Building the Problem
**********************

The Problem class serves as the central framework for defining inverse kinematics scenarios:

.. code-block:: python
   
   problem = Problem(mjx_model)

MJINX uses a component-based architecture where each component represents a mathematical function with specific semantics. All components share common attributes: a unique identifier, a gain parameter (weight in the optimization), and an optional mask for dimensional selection.

^^^^^^^^^^^^^^^
Task Components
^^^^^^^^^^^^^^^

Tasks define desired behaviors through objective functions that the solver attempts to minimize. In mathematical terms, a task represents a function :math:`f(q, t)` that maps from configuration space to a task-specific error measure.

Tasks are weighted through two parameters:

1. ``gain`` - Weight of the function in the objective (common across all components)
2. ``cost`` - Weight in velocity space (specific to the LocalIKSolver)

For example, to position an end-effector at a specific location:

.. code-block:: python
   
   frame_task = FrameTask(name="ee_task", cost=1, gain=20, body_name="link7")
   problem.add_component(frame_task)

You can also add regularization to minimize joint movement:

.. code-block:: python

   joint_task = JointTask("regularization", cost=0.1, gain=0)
   problem.add_component(joint_task)

^^^^^^^^^^^^^^^^^^
Barrier Components
^^^^^^^^^^^^^^^^^^

Barriers enforce constraints by defining functions that must remain positive: :math:`h(q, t) > 0`. These create boundaries in configuration space that the solver must respect.

For instance, to enforce joint limits:

.. code-block:: python

   joints_barrier = JointBarrier("jnt_barrier", gain=10)
   problem.add_component(joints_barrier)

After defining all components, compile the problem:

.. code-block:: python

   problem_data = problem.compile()

This compilation step transforms the high-level component specifications into optimized computational representations. Recompilation is necessary whenever component parameters change (e.g., updating a target position).

*********************
Solving the Problem
*********************

MJINX provides multiple solver implementations for different scenarios:

.. code-block:: python

   solver = LocalIKSolver(mjx_model)
   solver_data = solver.init(q)

The solver maintains internal state in ``solver_data``, which can include information like previous solutions for warm-starting.

To solve the problem:

.. code-block:: python

   opt_solution, solver_data = solver.solve(q, solver_data, problem_data)

The solution contains the optimal joint velocities (``v_opt``) and solver-specific information such as convergence status and error metrics.

To advance the system state using the computed velocities:

.. code-block:: python

   q = mjinx.configuration.integrate(
      mjx_model,
      q,
      velocity=opt_solution.v_opt,
      dt=dt,
   )

*********************
JAX Acceleration
*********************

MJINX leverages JAX's transformations to achieve significant performance improvements:

**JIT Compilation**

.. code-block:: python

   solve_jit = jax.jit(solver.solve)
   integrate_jit = jax.jit(integrate, static_argnames=["dt"])

**Vectorization for Batch Processing**

MJINX supports automatic vectorization for parallel computation of multiple IK problems:

.. code-block:: python

   # Vectorize initialization
   solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

   # Create template problem data with vmap dimensions
   with problem.set_vmap_dimension() as empty_problem_data:
      empty_problem_data.components["ee_task"].target_frame = 0

   # Vectorize solving and integration
   solve_jit = jax.jit(
      jax.vmap(
         solver.solve,
         in_axes=(0, 0, empty_problem_data),
      )
   )
   integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)))

This vectorization capability enables efficient parallel computation of multiple trajectories or configurations simultaneously, significantly accelerating optimization for complex robotics applications.

*********
Examples
*********

For more practical applications, explore the examples in the MJINX repository:

1. ``Kuka iiwa`` local inverse kinematics (single item and vectorized trajectory tracking)
2. ``Kuka iiwa`` global inverse kinematics (single item and vectorized trajectory tracking)
3. ``Go2`` quadruped robot batched squats example