import unittest

import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.barriers import JointBarrier
from mjinx.components.tasks import ComTask
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver
from mjinx.solvers._local_ik import OSQPParameters


class TestLocalIK(unittest.TestCase):
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

    def test_solver_parameters(self):
        """Testing that all solver parameters from datatyping are available"""
        osqp_params = OSQPParameters(
            check_primal_dual_infeasability=True,
            sigma=0.1,
            momentum=0.1,
            eq_qp_solve="cg",
            rho_start=0.1,
            rho_min=0.0,
            rho_max=1.0,
            stepsize_updates_frequency=500,
            primal_infeasible_tol=1e-3,
            dual_infeasible_tol=1e-3,
            maxiter=10,
            tol=1e-5,
            termination_check_frequency=2,
            implicit_diff_solve=lambda x: x,
        )
        _ = LocalIKSolver(self.model, **osqp_params)

        with self.assertRaises(TypeError):
            _ = LocalIKSolver(self.model, some_random_parameter="lol")

    def test_init(self):
        """Testing planner initialization"""

        solver = LocalIKSolver(model=self.model)
        data = solver.init()

        np.testing.assert_array_equal(data.v_prev, np.zeros(self.model.nv))

        with self.assertRaises(ValueError):
            _ = solver.init(np.arange(self.model.nv + 1))

        nonempty_data = solver.init(np.arange(self.model.nv))
        np.testing.assert_array_equal(nonempty_data.v_prev, np.arange(self.model.nv))

    def test_solve(self):
        """Testing solving functions"""
        solver = LocalIKSolver(model=self.model)
        solver_data = solver.init()

        with self.assertRaises(ValueError):
            solver.solve(np.ones(self.model.nq + 1), solver_data, self.problem_data)

        new_solution, new_solver_data = solver.solve(np.ones(self.model.nq), solver_data, self.problem_data)
        np.testing.assert_almost_equal(new_solution.v_opt, new_solver_data.v_prev)

        new_solution_from_data, _ = solver.solve_from_data(
            solver_data,
            self.problem_data,
            mjx.make_data(self.model).replace(qpos=np.ones(self.model.nq)),
        )

        np.testing.assert_almost_equal(new_solution.v_opt, new_solution_from_data.v_opt, decimal=3)
