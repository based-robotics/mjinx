import os
from datetime import datetime
from time import perf_counter

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from optax import adam
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier, PositionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.configuration import integrate
from mjinx.problem import Problem
from mjinx.solvers import GlobalIKSolver

# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mj_data = mj.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

# === Mjinx ===

# --- Constructing the problem ---
problem = Problem(mjx_model)

# Creating components of interest and adding them to the problem
frame_task = FrameTask("ee_task", cost=1, gain=20, obj_name="link7")
position_barrier = PositionBarrier(
    "ee_barrier",
    gain=0.1,
    obj_name="link7",
    limit_type="max",
    p_max=0.4,
    safe_displacement_gain=1e-2,
    mask=[1, 0, 0],
)
joints_barrier = JointBarrier("jnt_range", gain=0.1)

problem.add_component(frame_task)
problem.add_component(position_barrier)
problem.add_component(joints_barrier)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = GlobalIKSolver(mjx_model, adam(learning_rate=1e-2), dt=1e-2)

# Initializing initial condition
N_batch = 10000
np.random.seed(42)
q0 = jnp.array(
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
q = jnp.array(
    [
        np.clip(
            q0
            + np.random.uniform(
                -0.1,
                0.1,
                size=(mj_model.nq),
            ),
            q_min + 1e-1,
            q_max - 1e-1,
        )
        for _ in range(N_batch)
    ]
)

# --- Batching ---
solver_data = jax.vmap(solver.init, in_axes=0)(q)
with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["ee_task"].target_frame = 0

solve_jit = jax.jit(jax.vmap(solver.solve, in_axes=(0, 0, empty_problem_data)))
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])

# === Control loop ===
dt = 1e-2
ts = np.arange(0, 10, dt)

metrics: dict[str, list] = {
    "update_time_ms": [],
    "joint_positions": [],
    "joint_velocities": [],
    "target_frames": [],
    "optimization_steps": [],  # Track number of optimization steps
}

try:
    for t in ts:
        # Changing desired values
        target_frames = np.array(
            [
                [
                    0.4 + 0.3 * np.sin(t + 2 * np.pi * i / N_batch),
                    0.2,
                    0.4 + 0.3 * np.cos(t + 2 * np.pi * i / N_batch),
                    1,
                    0,
                    0,
                    0,
                ]
                for i in range(N_batch)
            ]
        )
        frame_task.target_frame = target_frames
        problem_data = problem.compile()

        # Solving the instance of the problem
        t0 = perf_counter()
        opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
        t1 = perf_counter()

        # Update positions directly for global IK
        q = opt_solution.q_opt

        # Logging
        t_update_ms = (t1 - t0) * 1e3

        # Log metrics
        metrics["update_time_ms"].append(t_update_ms)
        metrics["joint_positions"].append(np.array(q))
        metrics["joint_velocities"].append(opt_solution.v_opt)
        metrics["target_frames"].append(target_frames)
        metrics["optimization_steps"].append(3)  # Fixed number of steps used

        print(f"t={t:.2f}, Update time: {t_update_ms:.2f}ms")

except KeyboardInterrupt:
    print("Finalizing the simulation as requested...")
except Exception as e:
    print(e)

# Calculate and print summary statistics
compilation_time = metrics["update_time_ms"][0]
mean_update_time = np.mean(metrics["update_time_ms"][1:])
std_update_time = np.std(metrics["update_time_ms"][1:])
print("Benchmark Summary:")
print(f"Mean update time: {mean_update_time:.2f} ± {std_update_time:.2f} ms")

# Save data
save_data = {
    "timestamps": ts,
    "compilation_time": compilation_time,
    "n_batch": N_batch,
    "mean_update_time": mean_update_time,
    "std_update_time": std_update_time,
    "learning_rate": 1e-2,  # Save optimizer parameters
}

# Add component values and metrics to save data
for metric_name, values in metrics.items():
    save_data[metric_name] = np.array(values)

# Save to npz file
timestamp = int(perf_counter())

# Save to npz file
timestamp = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
script_dir = os.path.abspath(os.path.dirname(__file__))

filename = f"{script_dir}/logs/local_ik_benchmark_{timestamp}.npz"
np.savez_compressed(filename, **save_data)
print(f"\nBenchmark data saved to: {filename}")
