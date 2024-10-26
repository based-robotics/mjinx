import os
from datetime import datetime
from time import perf_counter

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier, PositionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.configuration import integrate
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver

# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mj_data = mj.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

# === Mjinx ===

# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-5, v_max=5)

# Creating components of interest and adding them to the problem
frame_task = FrameTask("ee_task", cost=1, gain=20, obj_name="link7")
position_barrier = PositionBarrier(
    "ee_barrier",
    gain=100,
    obj_name="link7",
    limit_type="max",
    p_max=0.4,
    safe_displacement_gain=1e-2,
    mask=[1, 0, 0],
)
joints_barrier = JointBarrier("jnt_range", gain=10)

problem.add_component(frame_task)
problem.add_component(position_barrier)
problem.add_component(joints_barrier)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = LocalIKSolver(mjx_model, maxiter=20)

# Initializing initial condition
N_batch = 10000
q0 = np.array(
    [
        -1.5878328,
        -2.0968683,
        -1.4339591,
        1.6550868,
        2.1080072,
        1.646142,
        -2.982619,
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
solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))
with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["ee_task"].target_frame = 0
solve_jit = jax.jit(
    jax.vmap(
        solver.solve,
        in_axes=(0, 0, empty_problem_data),
    )
)
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])


# === Control loop ===
dt = 1e-2
ts = np.arange(0, 10, dt)

metrics: dict[str, list] = {
    "update_time_ms": [],
    "joint_positions": [],
    "joint_velocities": [],
    "target_frames": [],
    "solver_iterations": [],
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

        # Integrating
        q = integrate_jit(
            mjx_model,
            q,
            opt_solution.v_opt,
            dt,
        )

        # Logging
        t_update_ms = (t1 - t0) * 1e3

        # Log metrics
        metrics["update_time_ms"].append(t_update_ms)
        metrics["joint_positions"].append(np.array(q))
        metrics["joint_velocities"].append(np.array(opt_solution.v_opt))
        metrics["target_frames"].append(np.array(target_frames).copy())
        metrics["solver_iterations"].append(np.array(opt_solution.iterations))

        print(f"t={t:.2f}, Update time: {t_update_ms:.2f}ms, Mean iterations: {np.mean(opt_solution.iterations)}")

except KeyboardInterrupt:
    print("Finalizing the simulation as requested...")
except Exception as e:
    print(e)

# Calculate and print summary statistics
compilation_time = metrics["update_time_ms"][0]
mean_update_time = np.mean(metrics["update_time_ms"][1:])
std_update_time = np.std(metrics["update_time_ms"][1:])
mean_iterations = np.mean(metrics["solver_iterations"])

print()
print("Benchmark Summary:")
print(f"Mean update time: {mean_update_time:.2f} ± {std_update_time:.2f} ms")
print(f"Mean solver iterations: {mean_iterations:.2f}")

# Save data
save_data = {
    "timestamps": ts,
    "n_batch": N_batch,
    "compilation_time": compilation_time,
    "mean_update_time": mean_update_time,
    "std_update_time": std_update_time,
    "mean_iterations": mean_iterations,
}

# Add component values and metrics to save data
for metric_name, values in metrics.items():
    save_data[metric_name] = np.array(values)

# Save to npz file
timestamp = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
script_dir = os.path.abspath(os.path.dirname(__file__))

filename = f"{script_dir}/logs/local_ik_vmapped_benchmark_{timestamp}.npz"
np.savez_compressed(filename, **save_data)
print(f"\nBenchmark data saved to: {filename}")
