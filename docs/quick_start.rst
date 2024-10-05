:github_url: https://github.com/based-robotics/mjinx/tree/docs/github_pages/docs/introduction.rst

************
Introduction
************

Inverse kinematics (IK) is the problem of finding desired joint configurations given a desired system state. `mjinx` simplifies the task of defining the desired (and undesired) system state using `Components`, and provides variety of solvers to solve the resulting problem.

The core ideas would be demonstrated using a 7 DoF manipulator `Kuka iiwa 14`

... code:: python
   
   from robot_descriptions.iiwa14_mj_description import MJCF_PATH 
   mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
   mjx_model = mjx.put_model(mj_model)
   ee_name = "link7"

.. image:: img/kuka_iiwa_14.png
   :alt: Kuka iiwa 14 manipulator

Building the problem
====================

To start composing the problem, you need to create its instance, which takes a `mujoco.mjx.Model`:
... code:: python
   
   problem = Problem(mjx_model)

Each problem is composed via `Components`. They always represent a function, but those functions have different meaning and purpose for each of component. However, all of them has name, gain (a.k.a weight in the problem), and mask to select desirable elements among all that component might include. For the details, see :class:`mjinx.component.Component`. Let's see, which particular components are present, and how to control our manipulator using them.

Task component
^^^^^^^^^^^^^^
Let's start with the _desired behavior_. In `mjinx` it could be specified using notion of :class:`mjinx.components.tasks.Task`. `Task` represents a non-negative function of state and time $f(q, t)$, which controller should keep as close to $0$ as possible. The weight of the tasks are specified by two parameters:
   1. `gain` -- weight of the function $f$ itself. It's commmon for all the components.
   2. `cost` -- weight of the residual in velocity space, used only by :class:`mjinx.solvers.LocalIKSolver`.

Suppose, you want to solve a task of moving an end-effector in a desired position. In `mjinx`, you can do it by adding a :class:`mjinx.components.tasks.FrameTask` which is defined for the end-effector:

... code:: python
   
   frame_task = FrameTask(name="ee_task", cost=1, gain=20, body_name=ee_name)
   problem.add_component(frame_task)

Simple, isn't it? However, it is possible to specify the desired behaviour even futher. For example, it would be nice if robot  would move as little as possible. Then, it's possible to add joint regularization task :class:`mjinx.components.JointTask`:

... code:: python
   joint_task = JointTask("regularization", cost=1e-1, gain=0)
   problem.add_component(joint_task)

Barrier component
^^^^^^^^^^^^^^^^^

Aprat from specifying _desired_ behavior, usually it's necessary to avoid _undesired_ one: the one, which should never occur under any circumstances. `mjinx` handles it by introducing :class:`mjinx.components.barriers.Barrier`. `Barrier` represents a function `h(q, t)`, that has to be strictly greater than zero: `h(q, t) > 0`. It's weight is specified by only `gain` parameter.

For example, very natural thing to ask -- don't break joint limits. The :class:`mjinx.components.barriers.JointBarrier` could handle this:

... code:: python
   
   joints_barrier = JointBarrier("jnt_barrier", gain=10)
   problem.add_component(joints_barrier)


Okay, the problem is ready. Let's check how we can solver it!


Solving the problem
===================

Skip for now






