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
from mjinx.visualize import BatchVisualizer

# === Mujoco ===

mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mj_data = mj.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()


# --- Mujoco visualization ---
# Initialize render window and launch it at the background
vis = BatchVisualizer(MJCF_PATH, n_models=5, alpha=0.5, record=False)

# Initialize a sphere marker for end-effector task
vis.add_markers(
    name=[f"ee_marker_{i}" for i in range(vis.n_models)],
    size=0.05,
    marker_alpha=0.5,
    color_begin=np.array([0, 1.0, 0.53]),
    color_end=np.array([0.38, 0.94, 1.0]),
    n_markers=vis.n_models,
)
vis.add_markers(
    name="blocking_plane",
    marker_type=mj.mjtGeom.mjGEOM_PLANE,
    size=np.array([0.5, 0.5, 0.02]),
    marker_alpha=0.7,
    color_begin=np.array([1, 0, 0]),
)

# === Mjinx ===

# --- Constructing the problem ---
# Creating problem formulation
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
# Set plane coodinate same to limiting one
vis.marker_data["blocking_plane"].pos = np.array([0.4, 0, 0.3])
vis.marker_data["blocking_plane"].rot = np.array(
    [
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0],
    ]
)

problem.add_component(frame_task)
problem.add_component(position_barrier)
problem.add_component(joints_barrier)

# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
solver = GlobalIKSolver(mjx_model, adam(learning_rate=1e-2), dt=1e-2)

# Initializing initial condition
N_batch = 100
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
# First of all, data should be created via vmapped init function
solver_data = jax.vmap(solver.init, in_axes=0)(q)

# To create a batch w.r.t. desired component's attributes, library defines convinient wrapper
# That sets all elements to None and allows user to mutate dataclasses of interest.
# After exiting the Context Manager, you'll get immutable jax dataclass object.
with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["ee_task"].target_frame = 0

# Vmapping solve and integrate functions.
# Note that for batching w.r.t. q both q and solver_data should be batched.
# Other approaches might work, but it would be undefined behaviour, please stick to this format.
solve_jit = jax.jit(jax.vmap(solver.solve, in_axes=(0, 0, empty_problem_data)))
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])


# === Control loop ===
dt = 1e-2
ts = np.arange(0, 20, dt)

try:
    for t in ts:
        # Changing desired values
        frame_task.target_frame = np.array(
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
        # After changes, recompiling the model
        problem_data = problem.compile()

        # Solving the instance of the problem
        for _ in range(3):
            opt_solution, solver_data = solve_jit(q, solver_data, problem_data)

        # Two options for retriving q:
        # Option 1, integrating:
        # q = integrate(mjx_model, q, opt_solution.v_opt, dt=dt)
        # Option 2, direct:
        q = opt_solution.q_opt

        # --- MuJoCo visualization ---
        for i, q_i in enumerate(frame_task.target_frame.wxyz_xyz[:: N_batch // vis.n_models, -3:]):
            vis.marker_data[f"ee_marker_{i}"].pos = q_i
        vis.update(q[:: N_batch // vis.n_models])

        # --- Joint limits ---
        for i in range(mj_model.nq):
            print(f"    Joint {i + 1}: {q_min[i]:0.2f} < {q[0, i]:0.2f} < {q_max[i]:0.2f}")
        print("-" * 80)

except KeyboardInterrupt:
    print("Finalizing the simulation as requested...")
except Exception as e:
    print(e)
finally:
    if vis.record:
        vis.save_video(round(1 / dt))
    vis.close()
