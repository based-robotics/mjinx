import traceback
from time import perf_counter

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3
from robot_descriptions.cassie_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier
from mjinx.components.constraints import ModelEqualityConstraint
from mjinx.components.tasks import ComTask, FrameTask
from mjinx.configuration import integrate, update
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver
from mjinx.visualize import BatchVisualizer

# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

# === Mjinx ===
problem = Problem(mjx_model, v_min=-1, v_max=1)

joints_barrier = JointBarrier("jnt_range", gain=0.1)
com_task = ComTask("com_task", cost=10.0, gain=50.0, mask=[1, 1, 1])
torso_task = FrameTask("torso_task", cost=1.0, gain=2.0, obj_name="cassie-pelvis", mask=[0, 0, 0, 1, 1, 1])
# Feet (in stance)
left_foot_task = FrameTask("left_foot_task", cost=20.0, gain=10.0, obj_name="left-foot", mask=[1, 1, 1, 1, 0, 1])
right_foot_task = FrameTask("right_foot_task", cost=20.0, gain=10.0, obj_name="right-foot", mask=[1, 1, 1, 1, 0, 1])

model_equality_constraint = ModelEqualityConstraint()

problem.add_component(com_task)
problem.add_component(torso_task)
problem.add_component(joints_barrier)
problem.add_component(left_foot_task)
problem.add_component(right_foot_task)
problem.add_component(model_equality_constraint)
# Initializing solver and its initial state
solver = LocalIKSolver(mjx_model, maxiter=10)

# Initializing initial condition
N_batch = 128
q0 = mj_model.keyframe("home").qpos
q = jnp.tile(q0, (N_batch, 1))

mjx_data = update(mjx_model, mjx.make_data(mjx_model).replace(qpos=jnp.array(q0)))

com0 = np.array(mjx_data.subtree_com[mjx_model.body_rootid[0]])
com_task.target_com = com0

# Get torso orientation and set it as target
torso_id = mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_BODY, "cassie-pelvis")
torso_quat = mjx_data.xquat[torso_id]
torso_task.target_frame = np.concatenate([np.zeros(3), torso_quat])

left_foot_id = mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_BODY, "left-foot")
left_foot_pos = mjx_data.xpos[left_foot_id]
left_foot_quat = mjx_data.xquat[left_foot_id]
left_foot_task.target_frame = jnp.concatenate([left_foot_pos, left_foot_quat])

right_foot_id = mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_BODY, "right-foot")
right_foot_pos = mjx_data.xpos[right_foot_id]
right_foot_quat = mjx_data.xquat[right_foot_id]
right_foot_task.target_frame = jnp.concatenate([right_foot_pos, right_foot_quat])


def compute_com_target(i, t, com0):
    # Compute target: here only z is modulated using a sine squared
    return jnp.array([0.0, 0.0, com0[2] - 0.3 * jnp.sin(t + 2 * jnp.pi * i / N_batch + jnp.pi / 2) ** 2])


compute_com_target_vmapped = jax.jit(jax.vmap(compute_com_target, in_axes=(0, None, None)))

# Compiling the problem upon any parameters update
problem_data = problem.compile()
# --- Batching ---
print("Setting up batched computations...")
solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["com_task"].target_com = 0

# Vmapping solve and integrate functions.
solve_jit = jax.jit(
    jax.vmap(
        solver.solve,
        in_axes=(0, None, 0, empty_problem_data),
    )
)
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])

t_warmup = perf_counter()
print("Performing warmup calls...")
# Warmup iterations for JIT compilation
com_task.target_com = compute_com_target_vmapped(jnp.arange(N_batch), 0, com0)
problem_data = problem.compile()
opt_solution, _ = solve_jit(q, mjx_data, solver_data, problem_data)
q_warmup = integrate_jit(mjx_model, q, opt_solution.v_opt, 0)

t_warmup_duration = perf_counter() - t_warmup
print(f"Warmup completed in {t_warmup_duration:.3f} seconds")

# === Control loop ===
print("\n=== Starting main loop ===")
dt = 1e-2
ts = np.arange(0, 20, dt)

# Performance tracking lists
compile_times = []
solve_times = []
integrate_times = []
n_steps = 0

try:
    for t in ts:
        # Update target COM using our separate function
        com_task.target_com = np.array(compute_com_target_vmapped(jnp.arange(N_batch), t, com0))

        # Recompile the problem (and time it)
        t1 = perf_counter()
        problem_data = problem.compile()
        jax.block_until_ready(problem_data)
        t2 = perf_counter()
        compile_times.append(t2 - t1)

        # Solve the instance and measure time
        t1 = perf_counter()
        opt_solution, solver_data = solve_jit(q, mjx_data, solver_data, problem_data)
        jax.block_until_ready(opt_solution)
        t2 = perf_counter()
        solve_times.append(t2 - t1)

        # Integrate:
        t1 = perf_counter()
        q = integrate_jit(mjx_model, q, opt_solution.v_opt, dt)
        jax.block_until_ready(q)
        t2 = perf_counter()
        integrate_times.append(t2 - t1)

        # --- MuJoCo visualization ---
        n_steps += 1

except KeyboardInterrupt:
    print("Finalizing the simulation as requested...")
except Exception:
    print(traceback.format_exc())
finally:
    # Performance report
    print("\n=== Performance Report ===")
    print(f"Total steps completed: {n_steps}")
    if compile_times:
        avg_compile = sum(compile_times) / len(compile_times)
        std_compile = np.std(compile_times)
        print(f"Compile:  {avg_compile * 1000:8.3f} ± {std_compile * 1000:8.3f} ms")
    if solve_times:
        avg_solve = sum(solve_times) / len(solve_times)
        std_solve = np.std(solve_times)
        print(f"Solve:    {avg_solve * 1000:8.3f} ± {std_solve * 1000:8.3f} ms")
    if integrate_times:
        avg_integrate = sum(integrate_times) / len(integrate_times)
        std_integrate = np.std(integrate_times)
        print(f"Integrate:{avg_integrate * 1000:8.3f} ± {std_integrate * 1000:8.3f} ms")
    if compile_times and solve_times and integrate_times:
        avg_total = sum(t1 + t2 + t3 for t1, t2, t3 in zip(compile_times, solve_times, integrate_times)) / len(
            solve_times
        )
        print(f"\nAverage computation time per step: {avg_total * 1000:.3f} ms")
        print(f"Effective computation rate: {1 / avg_total:.1f} Hz")
