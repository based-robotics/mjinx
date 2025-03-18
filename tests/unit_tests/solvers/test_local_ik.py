import unittest

import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np

from mjinx.components.barriers import JointBarrier
from mjinx.components.tasks import ComTask
from mjinx.components.constraints import JaxConstraint
from mjinx.problem import Problem, JaxProblemData
from mjinx.solvers import LocalIKSolver
from mjinx.solvers._local_ik import OSQPParameters
from mjinx import configuration


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
        self.problem = Problem(self.model)
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
        solver = LocalIKSolver(
            model=self.model,
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
        data = solver.init()
        np.testing.assert_array_equal(data.v_prev, np.zeros(self.model.nv))

        with self.assertRaises(ValueError):
            _ = solver.init(np.arange(self.model.nv + 1))

        nonempty_data = solver.init(np.arange(self.model.nv))
        np.testing.assert_array_equal(nonempty_data.v_prev, np.arange(self.model.nv))

    def test_solve(self):
        """Testing solving functions"""
        solver = LocalIKSolver(
            model=self.model,
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
        solver_data = solver.init()

        with self.assertRaises(ValueError):
            solver.solve(np.ones(self.model.nq + 1), solver_data, self.problem_data)

        new_solution, new_solver_data = solver.solve(np.ones(self.model.nq), solver_data, self.problem_data)
        np.testing.assert_almost_equal(new_solution.v_opt, new_solver_data.v_prev)

        new_solution_from_data, _ = solver.solve_from_data(
            solver_data, self.problem_data, configuration.update(self.model, np.ones(self.model.nq))
        )
        np.testing.assert_almost_equal(new_solution.v_opt, new_solution_from_data.v_opt, decimal=3)

    def test_compute_qp_matrices_no_constraints(self):
        """
        Test __compute_qp_matrices in the case with no hard/soft components.
        This covers lines 268-276 where inequality constraints are combined.
        """
        solver = LocalIKSolver(
            self.model,
            check_primal_dual_infeasability=True,
            sigma=1e-6,
            momentum=1.6,
            eq_qp_solve="cg",
            rho_start=0.1,
            rho_min=1e-6,
            rho_max=1e6,
            stepsize_updates_frequency=10,
            primal_infeasible_tol=1e-4,
            dual_infeasible_tol=1e-4,
            maxiter=4000,
            tol=1e-3,
            termination_check_frequency=5,
            implicit_diff_solve=lambda A, b: jnp.linalg.solve(A, b),
        )
        # Create a dummy problem data with no additional components.
        dummy_problem_data = JaxProblemData(
            model=self.model,
            v_min=jnp.zeros(self.model.nv),
            v_max=jnp.ones(self.model.nv),
            components={},
        )
        model_data = mjx.make_data(self.model)
        H_total, c_total, A_mat, b_vec, G_mat, h_vec = solver._LocalIKSolver__compute_qp_matrices(
            dummy_problem_data, model_data
        )
        # Since no hard constraints exist, A_mat and b_vec should be None.
        self.assertIsNone(A_mat)
        self.assertIsNone(b_vec)
        # The velocity limit constraints are always added: 2*nv rows.
        self.assertEqual(G_mat.shape[0], 2 * self.model.nv)
        self.assertEqual(h_vec.shape[0], 2 * self.model.nv)

    def test_compute_qp_matrices_with_hard_constraint(self):
        """
        Test __compute_qp_matrices when a hard constraint is present.
        This helps cover the branch where hard constraints (lines 268-276) are processed.
        """

        # Define a dummy hard constraint component.
        class DummyHardConstraint(JaxConstraint):
            def compute_jacobian(self, model_data):
                # Return an identity matrix as a dummy jacobian.
                return jnp.eye(self.model.nv)

            def __call__(self, model_data):
                return jnp.zeros(self.model.nv)

        dummy_constraint = DummyHardConstraint(
            "dummy_hard_constraint",
            model=self.model,
            gain_fn=lambda x: x,
            active=True,
            mask_idxs=np.arange(self.model.nv),
            vector_gain=np.ones(self.model.nv),
            hard_constraint=True,
            soft_constraint_cost=np.ones(self.model.nv),
        )
        # Extend the problem data components with the dummy hard constraint.
        components = dict(self.problem_data.components)
        components["hard_constraint"] = dummy_constraint
        dummy_problem_data = JaxProblemData(
            model=self.model,
            v_min=jnp.zeros(self.model.nv),
            v_max=jnp.ones(self.model.nv),
            components=components,
        )
        model_data = mjx.make_data(self.model)
        solver = LocalIKSolver(
            self.model,
            check_primal_dual_infeasability=True,
            sigma=1e-6,
            momentum=1.6,
            eq_qp_solve="cg",
            rho_start=0.1,
            rho_min=1e-6,
            rho_max=1e6,
            stepsize_updates_frequency=10,
            primal_infeasible_tol=1e-4,
            dual_infeasible_tol=1e-4,
            maxiter=4000,
            tol=1e-3,
            termination_check_frequency=5,
            implicit_diff_solve=lambda A, b: jnp.linalg.solve(A, b),
        )
        H_total, c_total, A_mat, b_vec, G_mat, h_vec = solver._LocalIKSolver__compute_qp_matrices(
            dummy_problem_data, model_data
        )
        # Now, A_mat and b_vec should be provided by the hard constraint.
        self.assertIsNotNone(A_mat)
        self.assertIsNotNone(b_vec)
        self.assertEqual(A_mat.shape, (self.model.nv, self.model.nv))
        self.assertEqual(b_vec.shape, (self.model.nv,))

    def test_solve_from_data_dummy(self):
        """
        Test solve_from_data by monkey-patching the OSQP run function.
        This covers lines 279-284 where QP matrices are passed and a dummy solution is returned.
        """
        solver = LocalIKSolver(
            self.model,
            check_primal_dual_infeasability=True,
            sigma=1e-6,
            momentum=1.6,
            eq_qp_solve="cg",
            rho_start=0.1,
            rho_min=1e-6,
            rho_max=1e6,
            stepsize_updates_frequency=10,
            primal_infeasible_tol=1e-4,
            dual_infeasible_tol=1e-4,
            maxiter=4000,
            tol=1e-3,
            termination_check_frequency=5,
            implicit_diff_solve=lambda A, b: jnp.linalg.solve(A, b),
        )

        # Define a dummy solution class to simulate OSQP run output.
        class DummySolution:
            class Params:
                def __init__(self, primal, dual_eq, dual_ineq):
                    self.primal = primal
                    self.dual_eq = dual_eq
                    self.dual_ineq = dual_ineq

            class State:
                def __init__(self, iter_num, error, status):
                    self.iter_num = iter_num
                    self.error = error
                    self.status = status

            def __init__(self, primal):
                self.params = DummySolution.Params(primal, jnp.array(0), jnp.array(0))
                self.state = DummySolution.State(iter_num=5, error=0.001, status=1)

        def dummy_run(*args, **kwargs):
            return DummySolution(primal=jnp.ones(solver.model.nv))

        solver._solver.run = dummy_run

        dummy_problem_data = JaxProblemData(
            model=self.model,
            v_min=jnp.zeros(self.model.nv),
            v_max=jnp.ones(self.model.nv),
            components={},
        )
        model_data = mjx.make_data(self.model)
        solution, solver_data = solver.solve_from_data(solver.init(), dummy_problem_data, model_data)
        self.assertTrue(jnp.allclose(solution.v_opt, jnp.ones(self.model.nv)))
        self.assertTrue(jnp.allclose(solver_data.v_prev, jnp.ones(self.model.nv)))

    def test_init_valid_dimension(self):
        """
        Test that init correctly accepts valid velocity dimensions.
        This covers the branch in init (lines 314-322).
        """
        solver = LocalIKSolver(
            self.model,
            check_primal_dual_infeasability=True,
            sigma=1e-6,
            momentum=1.6,
            eq_qp_solve="cg",
            rho_start=0.1,
            rho_min=1e-6,
            rho_max=1e6,
            stepsize_updates_frequency=10,
            primal_infeasible_tol=1e-4,
            dual_infeasible_tol=1e-4,
            maxiter=4000,
            tol=1e-3,
            termination_check_frequency=5,
            implicit_diff_solve=lambda A, b: jnp.linalg.solve(A, b),
        )
        valid_v = jnp.arange(self.model.nv)
        data = solver.init(valid_v)
        self.assertEqual(data.v_prev.shape, (self.model.nv,))


if __name__ == "__main__":
    unittest.main()
