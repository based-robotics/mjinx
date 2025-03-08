import os
from datetime import datetime
from time import perf_counter

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier, PositionBarrier, SelfCollisionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.configuration import integrate
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver

# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mj_data = mj.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)

# === Mjinx ===
problem = Problem(mjx_model, v_min=-100, v_max=100)

frame_task = FrameTask("ee_task", cost=1, gain=20, obj_name="link7")
position_barrier = PositionBarrier(
    "ee_barrier",
    gain=100,
    obj_name="link7",
    limit_type="max",
    p_max=0.3,
    safe_displacement_gain=1e-2,
    mask=[1, 0, 0],
)
joints_barrier = JointBarrier("jnt_range", gain=10)
self_collision_barrier = SelfCollisionBarrier(
    "self_collision_barrier",
    gain=1.0,
    d_min=0.01,
)

problem.add_component(frame_task)
problem.add_component(position_barrier)
problem.add_component(joints_barrier)
problem.add_component(self_collision_barrier)

problem_data = problem.compile()

solver = LocalIKSolver(mjx_model, maxiter=20)

q = jnp.array(
    [
        -1.4238753,
        -1.7268502,
        -0.84355015,
        2.0962472,
        2.1339328,
        2.0837479,
        -2.5521986,
    ]
)
solver_data = solver.init()

solve_jit = jax.jit(solver.solve)
integrate_jit = jax.jit(integrate, static_argnames=["dt"])


# === Control loop ===
dt = 1e-2
ts = np.arange(0, 10, dt)

# Additional metrics for comprehensive benchmarking
metrics: dict[str, list] = {
    "update_time_ms": [],
    "joint_positions": [],
    "joint_velocities": [],
    "target_frames": [],
    "solver_iterations": [],
}

for t in ts:
    # Changing desired values
    target_frame = np.array([0.2 + 0.2 * jnp.sin(t) ** 2, 0.2, 0.2, 1, 0, 0, 0])
    frame_task.target_frame = target_frame
    # After changes, recompiling the model
    problem_data = problem.compile()

    # Solving the instance of the problem
    t0 = perf_counter()
    opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
    t1 = perf_counter()

    # Integrating
    q = integrate_jit(
        mjx_model,
        q,
        velocity=opt_solution.v_opt,
        dt=dt,
    )

    # Logging
    t_update_ms = (t1 - t0) * 1e3

    # Log additional metrics
    metrics["update_time_ms"].append(t_update_ms)
    metrics["joint_positions"].append(np.array(q.copy()))
    metrics["joint_velocities"].append(np.array(opt_solution.v_opt.copy()))
    metrics["target_frames"].append(target_frame)
    metrics["solver_iterations"].append(opt_solution.iterations)

    print(f"t={t:.2f}, Update time: {t_update_ms:.2f}ms, Iterations: {opt_solution.iterations}")

# Calculate and print summary statistics
compilation_time = metrics["update_time_ms"][0]
mean_update_time = np.mean(metrics["update_time_ms"][1:])
std_update_time = np.std(metrics["update_time_ms"][1:])
mean_iterations = np.mean(metrics["solver_iterations"])
print()
print("Benchmark Summary:")
print(f"Mean update time: {mean_update_time:.2f} ± {std_update_time:.2f} ms")
print(f"Mean solver iterations: {mean_iterations:.2f}")

# Convert lists to numpy arrays for saving
save_data = {
    "timestamps": ts,
    "compilation_time": compilation_time,
    "mean_update_time": mean_update_time,
    "std_update_time": std_update_time,
    "mean_iterations": mean_iterations,
}


# Add metrics to save data
for metric_name, values in metrics.items():
    save_data[metric_name] = np.array(values)

# Save to npz file
timestamp = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
script_dir = os.path.abspath(os.path.dirname(__file__))

filename = f"{script_dir}/logs/local_ik_benchmark_{timestamp}.npz"
np.savez_compressed(filename, **save_data)
print(f"\nBenchmark data saved to: {filename}")
