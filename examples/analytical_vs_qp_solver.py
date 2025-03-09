"""
Example comparing Analytical and QP solvers for inverse kinematics without barriers.

This example demonstrates the performance benefits of using the analytical solver
for problems without inequality constraints. It compares the analytical solver with
the OSQP solver for a simple inverse kinematics problem with only task objectives
and no barriers or inequality constraints.

The analytical solver directly computes the solution and clips it to satisfy velocity
limits, which is much faster than solving the full QP problem with the OSQP solver.
"""

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from robot_descriptions.iiwa14_mj_description import MJCF_PATH
from time import perf_counter

from mjinx.components.tasks import FrameTask, JointTask
from mjinx.configuration import integrate
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver

print("=== Initializing ===")

# === Mujoco ===
print("Loading MuJoCo model...")
mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mjx_model = mjx.put_model(mj_model)

# === Mjinx ===
print("Setting up optimization problem...")

# --- Problem: Only tasks (no barriers) ---
# Create a problem with only task components (no barriers)
problem = Problem(mjx_model, v_min=-50000, v_max=50000)

# Add a frame task for end-effector control
frame_task = FrameTask("ee_task", cost=1.0, gain=20, obj_name="link7")
problem.add_component(frame_task)

# Add a joint task with small weight for regularization
# This ensures the system is well-conditioned and has a unique solution
joint_task = JointTask("joint_reg", cost=0.05, gain=1.0)
# Set the target to the initial configuration to provide a "home" position
joint_task.target = np.zeros(mjx_model.nq)
problem.add_component(joint_task)

problem_data = problem.compile()

# --- Initializing solvers ---
print("Initializing solvers...")
# Create solvers with analytical solver enabled and disabled
solver_analytical = LocalIKSolver(mjx_model, use_analytical_solver=True)
solver_qp = LocalIKSolver(mjx_model, use_analytical_solver=False, maxiter=30, tol=1e-12)

# --- Initial configuration ---
# Use a small batch size for faster execution and easier debugging
N_batch = 1000
np.random.seed(42)
q0 = jnp.array([-1.4, -1.7, -0.8, 2.1, 2.1, 2.0, -2.5])
q = jnp.repeat(q0[None, :], N_batch, axis=0)

# Create copies of q for each solver
q_analytical = q.copy()
q_qp = q.copy()

# --- Batching ---
print("Setting up batched computations...")
# Initialize solver data for each solver
solver_data_analytical = jax.vmap(solver_analytical.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))
solver_data_qp = jax.vmap(solver_qp.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

# Set up vmapped problem data
with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["ee_task"].target_frame = 0
    # No need to set joint_task target as it's the same for all batch elements

# Vmapping solve and integrate functions for both solvers
solve_analytical_jit = jax.jit(jax.vmap(solver_analytical.solve, in_axes=(0, 0, empty_problem_data)))
solve_qp_jit = jax.jit(jax.vmap(solver_qp.solve, in_axes=(0, 0, empty_problem_data)))
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])

# === Warmup phase ===
print("\n=== Performing warmup ===")
# Create a single target frame for warmup
warmup_target_frames = np.array([[0.4, 0.2, 0.5 + 0.1 * i / N_batch, 1, 0, 0, 0] for i in range(N_batch)])

# Set target frames
frame_task.target_frame = warmup_target_frames
problem_data = problem.compile()

# Warmup analytical solver
print("Warming up analytical solver...")
opt_solution_analytical, _ = solve_analytical_jit(q.copy(), solver_data_analytical, problem_data)

# Warmup QP solver
print("Warming up QP solver...")
opt_solution_qp, _ = solve_qp_jit(q.copy(), solver_data_qp, problem_data)

print("Warmup completed. JIT compilation should now be finished.")

# === Performance comparison ===
print("\n=== Starting performance comparison ===")
dt = 2e-2
num_steps = 10  # Small number of steps for quick testing

# Reset configurations and solver data for actual performance measurement
q_init = q.copy()
solver_data_analytical = jax.vmap(solver_analytical.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))
solver_data_qp = jax.vmap(solver_qp.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

# Preallocate arrays for performance tracking
solve_times_analytical = np.zeros(num_steps)
solve_times_qp = np.zeros(num_steps)
v_diffs = np.zeros((num_steps, N_batch))
q_diffs = np.zeros((num_steps, N_batch))

# Main loop
for step in range(num_steps):
    # Generate simple target frames (fixed position with small variations)
    target_frames = np.array([[0.4, 0.2, 0.5 + 0.1 * i / N_batch, 1, 0, 0, 0] for i in range(N_batch)])
    frame_task.target_frame = target_frames
    problem_data = problem.compile()

    # Solve with analytical solver (will compute solution directly and clip to velocity limits)
    t1 = perf_counter()
    opt_solution_analytical, solver_data_analytical = solve_analytical_jit(q_init, solver_data_analytical, problem_data)
    t2 = perf_counter()
    solve_times_analytical[step] = t2 - t1

    # Solve with QP solver (will use OSQP to solve the full QP problem)
    t1 = perf_counter()
    opt_solution_qp, solver_data_qp = solve_qp_jit(q_init, solver_data_qp, problem_data)
    t2 = perf_counter()
    solve_times_qp[step] = t2 - t1

    # Compare velocity solutions
    v_diffs[step] = np.linalg.norm(np.array(opt_solution_analytical.v_opt) - np.array(opt_solution_qp.v_opt), axis=1)

    # Integrate solutions
    q_init = integrate_jit(mjx_model, q_init, opt_solution_analytical.v_opt, dt).copy()

    # print(f"v_analytical: {opt_solution_analytical.v_opt[0]}")
    # print(f"v_qp: {opt_solution_qp.v_opt[0]}")
    difference = opt_solution_analytical.v_opt - opt_solution_qp.v_opt
    print(20 * "=")
    print(f" Average difference in v across all batch elements: {np.linalg.norm(difference/N_batch)}")
    print(f" Maximum difference in v across all batch elements: {np.max(np.linalg.norm(difference, axis=1)/N_batch)}")
    print(f" Residual between analytical and qp solution for first batch element: {difference[0]}")
    print(f" Completed step {step + 1}/{num_steps}")

# Print performance report
print("\n=== Performance Report ===")
print(f"Total steps completed: {num_steps}")

print("\nComputation times per step (ms):")
avg_analytical = np.mean(solve_times_analytical)
std_analytical = np.std(solve_times_analytical)
print(f"Analytical solver: {avg_analytical*1000:8.3f} ± {std_analytical*1000:8.3f} ms")

avg_qp = np.mean(solve_times_qp)
std_qp = np.std(solve_times_qp)
print(f"QP solver       : {avg_qp*1000:8.3f} ± {std_qp*1000:8.3f} ms")

# Calculate speedup
speedup = avg_qp / avg_analytical
print(f"\nSpeedup: {speedup:.2f}x")

# Print solution comparison
print("\n=== Solution Comparison ===")

# Velocity solution differences
v_diffs_flat = v_diffs.flatten()
print("\nVelocity solution differences (L2 norm):")
print(f"  Average: {np.mean(v_diffs_flat):.6f}")
print(f"  Maximum: {np.max(v_diffs_flat):.6f}")
print(f"  Minimum: {np.min(v_diffs_flat):.6f}")
