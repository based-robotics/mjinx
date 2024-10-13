import time
from os.path import join

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from robot_descriptions.go2_mj_description import PACKAGE_PATH

from mjinx.components.barriers import JointBarrier
from mjinx.components.tasks import ComTask, FrameTask
from mjinx.configuration import integrate, update
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver

MJCF_PATH = join(PACKAGE_PATH, "go2_mjx.xml")
# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

# --- Mujoco visualization ---
# Initialize render window and launch it at the background
# vis = BatchVisualizer(MJCF_PATH, n_models=1, alpha=1.0)
# vis.add_markers(
#     name="com_marker",
#     size=0.05,
#     marker_alpha=0.4,
#     color_begin=np.array([0, 1.0, 0.53]),
# )

# === Mjinx ===
# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-5, v_max=5)

# Creating components of interest and adding them to the problem
com_task = ComTask("com_task", cost=1.0, gain=5)
frame_task = FrameTask("body_orientation_task", cost=1.0, gain=10, body_name="base", mask=[0, 0, 0, 1, 1, 1])
joints_barrier = JointBarrier("jnt_range", gain=0.05, floating_base=True)

problem.add_component(com_task)
problem.add_component(frame_task)
problem.add_component(joints_barrier)

for foot in ["FL", "FR", "RL", "RR"]:
    task = FrameTask(
        foot + "_foot_task",
        body_name=foot,
        cost=10.0,
        gain=5.0,
        mask=[1, 1, 1, 0, 0, 0],
    )
    problem.add_component(task)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = LocalIKSolver(mjx_model, maxiter=5)

# Initializing initial condition
q = jnp.array(
    [
        -0.0,  # Torso position
        0.0,
        0.3,
        1.0,  # Torso orientation
        0.0,
        0.0,
        0.0,
        0.0,  # Front Left foot
        0.8,
        -1.57,
        0.0,  # Front Right foot
        0.8,
        -1.57,
        0.0,  # Rare Left foot
        0.8,
        -1.57,
        0.0,  # Rare Right foot
        0.8,
        -1.57,
    ]
)

mjx_data = update(mjx_model, jnp.array(q))

for foot in ["FL", "FR", "RL", "RR"]:
    foot_pos = mjx_data.xpos[mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_BODY, foot)]
    print(foot_pos)
    problem.component(foot + "_foot_task").target_frame = jnp.array([*foot_pos, 1, 0, 0, 0])

solver_data = solver.init()
# --- Batching ---
solve_jit = jax.jit(solver.solve)
integrate_jit = jax.jit(integrate)

# === Control loop ===
dt = 1e-2
ts = np.arange(0, 20, dt)

t_solve_avg = 0.0
n = 0


print()

# try:
for t in ts:
    # Changing desired values
    # frame_task.target_frame = SE3.from_rotation_and_translation(
    #     SO3.from_matrix(
    #         np.array(
    #             [
    #                 [
    #                     [np.cos(2 * t + delta_phase(i)), 0, np.sin(2 * t + delta_phase(i))],
    #                     [0, 1, 0],
    #                     [-np.sin(2 * t + delta_phase(i)), 0, np.cos(2 * t + delta_phase(i))],
    #                 ]
    #                 for i in range(N_batch)
    #             ]
    #         )
    #     ),
    #     np.zeros((N_batch, 3)),
    # )
    com_task.target_com = jnp.array([0.0, 0.0, 0.2 + 0.1 * np.sin(t + np.pi / 2)])
    # After changes, recompiling the model
    problem_data = problem.compile()
    t0 = time.perf_counter()

    # Solving the instance of the problem
    opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
    t1 = time.perf_counter()

    # Integrating
    q = integrate_jit(
        mjx_model,
        q,
        opt_solution.v_opt,
        dt,
    )

    # --- MuJoCo visualization ---
    # vis.marker_data["com_marker"].pos = np.array(com_task.target_com)
    # vis.update(q)

    t2 = time.perf_counter()
    t_solve = (t1 - t0) * 1e3
    t_interpolate = (t2 - t1) * 1e3
    print(f"Solve time: {t_solve:0.2f}ms, interpolation time: {t_interpolate:0.2f}")
    if t > 0:
        t_solve_avg = t_solve_avg + (t_solve - t_solve_avg) / (n + 1)
        n += 1
# except KeyboardInterrupt:
#     print("Finalizing the simulation as requested...")
# except Exception as e:
#     print(e)
# finally:
#     # if vis.record:
#     #     vis.save_video(round(1 / dt))
#     # vis.close()
#     print(f"Avg solving time: {t_solve_avg:0.3f}ms")
