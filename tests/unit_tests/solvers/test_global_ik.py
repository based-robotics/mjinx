import unittest

import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
import optax

from mjinx.components.barriers import JointBarrier
from mjinx.components.tasks import ComTask
from mjinx.problem import Problem
from mjinx.solvers import GlobalIKSolver
from mjinx.solvers._local_ik import OSQPParameters


class TestGlobalIK(unittest.TestCase):
    def setUp(self):
        """Setting up basic model, components, and problem data."""

        self.model = mjx.put_model(
            mj.MjModel.from_xml_string(
                """
        <mujoco>
            <worldbody>
                <body name="body1">
                    <joint name="jnt1" type="hinge" axis="1 -1 0"/>
                    <geom name="box1" size=".3"/>
                    <geom name="box2" pos=".6 .6 .6" size=".3"/>
                </body>
            </worldbody>
        </mujoco>
        """
            )
        )
        self.dummy_task = ComTask("com_task", cost=1.0, gain=1.0)
        self.dummy_barrier = JointBarrier("joint_barrier", gain=1.0)
        self.problem = Problem(
            self.model,
            v_min=0,
            v_max=0,
        )
        self.problem.add_component(self.dummy_task)
        self.problem.add_component(self.dummy_barrier)

        self.problem_data = self.problem.compile()

    def test_init(self):
        """Testing planner initialization"""

        solver = GlobalIKSolver(model=self.model, optimizer=optax.adam(learning_rate=1e-3))

        with self.assertRaises(ValueError):
            _ = solver.init(q=np.arange(self.model.nq + 1))

        data = solver.init(q=np.arange(self.model.nq))

        # Check that it correctly initialized for chosen optimizer
        self.assertIsInstance(data.optax_state[0], optax.ScaleByAdamState)
        self.assertIsInstance(data.optax_state[1], optax.EmptyState)

    def test_loss_fn(self):
        self.problem.remove_component(self.dummy_barrier.name)
        self.dummy_task.target_com = [0.3, 0.3, 0.3]
        new_problem_data = self.problem.compile()

        solver = GlobalIKSolver(model=self.model, optimizer=optax.adam(learning_rate=1e-3))
        loss = solver.loss_fn(np.zeros(self.model.nq), problem_data=new_problem_data)

        self.assertEqual(loss, 0)

    def test_loss_grad(self):
        solver = GlobalIKSolver(model=self.model, optimizer=optax.adam(learning_rate=1e-3))
        grad = solver.grad_fn(np.zeros(self.model.nq), problem_data=self.problem_data)

        self.assertEqual(grad.shape, (self.model.nv,))

    def test_solve(self):
        """Testing solving functions"""
        solver = GlobalIKSolver(model=self.model, optimizer=optax.adam(learning_rate=1e-2), dt=1e-3)
        solver_data = solver.init(q=np.ones(self.model.nq))

        with self.assertRaises(ValueError):
            solver.solve(np.ones(self.model.nq + 1), solver_data, self.problem_data)

        new_solution, _ = solver.solve(np.ones(self.model.nq), solver_data, self.problem_data)

        new_solution_from_data, _ = solver.solve_from_data(
            solver_data,
            self.problem_data,
            mjx.make_data(self.model).replace(qpos=np.ones(self.model.nq)),
        )

        np.testing.assert_almost_equal(new_solution.v_opt, new_solution_from_data.v_opt, decimal=3)


unittest.main()
