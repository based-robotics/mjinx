:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/introduction.rst

============
Quick Start
============

Here's a quick start guide to get you started with MJINX.

You can also follow along with this quick start guide by clicking on |colab|. This interactive notebook provides a hands-on environment to experiment with MJINX without installing anything on your local machine.

   .. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/based-robotics/mjinx/blob/main/examples/notebooks/turoial.ipynb
      :alt: Open in Colab

****************
Complete Example
****************

Here's a complete example showing MJINX in action:

.. code-block:: python

    from mujoco import mjx
    from mjinx.problem import Problem

    # Initialize the robot model using MuJoCo
    MJCF_PATH = "path_to_mjcf.xml"
    mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
    mjx_model = mjx.put_model(mj_model)

    # Create instance of the problem
    problem = Problem(mjx_model)

    # Add tasks to track desired behavior
    frame_task = FrameTask("ee_task", cost=1, gain=20, body_name="link7")
    problem.add_component(frame_task)

    # Add barriers to keep robot in a safety set
    joints_barrier = JointBarrier("jnt_range", gain=10)
    problem.add_component(joints_barrier)

    # Initialize the solver
    solver = LocalIKSolver(mjx_model)

    # Initializing initial condition
    q0 = np.zeros(7)

    # Initialize solver data
    solver_data = solver.init()

    # jit-compiling solve and integrate 
    solve_jit = jax.jit(solver.solve)
    integrate_jit = jax.jit(integrate, static_argnames=["dt"])

    # === Control loop ===
    for t in np.arange(0, 5, 1e-2):
        # Changing problem and compiling it
        frame_task.target_frame = np.array([0.1 * np.sin(t), 0.1 * np.cos(t), 0.1, 1, 0, 0,])
        problem_data = problem.compile()

        # Solving the instance of the problem
        opt_solution, solver_data = solve_jit(q, solver_data, problem_data)

        # Integrating
        q = integrate_jit(
            mjx_model,
            q,
            opt_solution.v_opt,
            dt,
        )

Let's break this down step by step to understand how MJINX works.

**********************
Building the Problem
**********************

First, create an instance of the :class:`Problem <mjinx.problem.Problem>` class with your MuJoCo MJX model:

.. code-block:: python
   
   problem = Problem(mjx_model)

A problem consists of various *Components*, each representing a function with specific meaning and purpose. All components have a name, gain (weight in the optimization), and an optional mask to select specific elements. See :class:`Component <mjinx.components._base.Component>` for details.

^^^^^^^^^^^^^^^
Task Components
^^^^^^^^^^^^^^^

Tasks define your *desired behavior* - what you want your robot to achieve. In MJINX, a :class:`Task <mjinx.components.tasks._base.Task>` represents a function :math:`f(q, t)` that the controller tries to minimize (keep close to zero).

Task importance is specified by two parameters:

1. ``gain`` - Weight of the function itself (common across all components)
2. ``cost`` - Weight of the residual in velocity space (used by :class:`LocalIKSolver <mjinx.solvers._local_ik.LocalIKSolver>`)

To position an end-effector at a desired location, add a :class:`FrameTask <mjinx.components.tasks._obj_frame_task.FrameTask>`:

.. code-block:: python
   
   frame_task = FrameTask(name="ee_task", cost=1, gain=20, body_name="link7")
   problem.add_component(frame_task)

^^^^^^^^^^^^^^^^^^
Barrier Components
^^^^^^^^^^^^^^^^^^

Barriers define *constraints* - conditions that must never be violated. A :class:`Barrier <mjinx.components.barriers._base.Barrier>` represents a function :math:`h(q, t)` that must always remain positive: :math:`h(q, t) > 0`.

For example, to enforce joint limits, use a :class:`JointBarrier <mjinx.components.barriers._joint_barrier.JointBarrier>`:

.. code-block:: python

   joints_barrier = JointBarrier("jnt_barrier", gain=10)
   problem.add_component(joints_barrier)

When you've finished building your problem, compile it:

.. code-block:: python

   problem_data = problem.compile()

Compilation converts each :class:`Component <mjinx.components._base.Component>` into its corresponding :class:`JaxComponent <mjinx.components._base.JaxComponent>`. You must recompile whenever you modify a component (e.g., changing a target position).

*********************
Solving the Problem
*********************

^^^^^^^
Solvers
^^^^^^^

MJINX provides different solvers that inherit from the :class:`Solver <mjinx.solver._base.Solver>` class. Let's use the :class:`LocalIKSolver <mjinx.solver._local_ik.LocalIKSolver>`:

.. code-block:: python

   solver = LocalIKSolver(mjx_model, maxiter=20)
   solver_data = solver.init()

To solve the problem, provide the current state ``q``, the solver data, and the problem data:

.. code-block:: python

   opt_solution, solver_data = solver.solve(q, solver_data, problem_data)

The ``opt_solution`` contains the optimal joint velocity ``v_opt`` and may include additional information.

^^^^^^^^^^^^^^^^^^^^^^^
Configuration Utilities
^^^^^^^^^^^^^^^^^^^^^^^

Use :func:`mjinx.configuration.integrate <mjinx.configuration.integrate>` to advance the system state:

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

One of MJINX's key advantages is its JAX compatibility. All methods in the ``Solver`` class and ``configuration`` module can be accelerated using JAX transformations.

For performance, you can JIT-compile the solver and integration functions:

.. code-block:: python

   solve_jit = jax.jit(solver.solve)
   integrate_jit = jax.jit(mjinx.configuration.integrate)

You can even vectorize the computation to solve multiple problems in parallel:

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
   integrate_jit = jax.jit(jax.vmap(mjinx.configuration.integrate, in_axes=(None, 0, 0, None)))

This approach enables efficient parallel computation of multiple IK solutions, significantly accelerating your robotics applications.

*********
Examples
*********

For more practical examples, check out the examples directory in the MJINX repository:

1. ``Kuka iiwa`` local inverse kinematics (single item and vmapped over desired trajectory)
2. ``Kuka iiwa`` global inverse kinematics (single item and vmapped over desired trajectory)
3. ``Go2`` batched squats example