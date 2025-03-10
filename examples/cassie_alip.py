import traceback
from collections.abc import Callable

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from robot_descriptions.cassie_mj_description import MJCF_PATH

from mjinx.components.barriers import JointBarrier
from mjinx.components.constraints import ModelEqualityConstraint
from mjinx.components.tasks import ComTask, FrameTask
from mjinx.configuration import integrate, update
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver
from mjinx.visualize import BatchVisualizer


def make_dead_beat_alip(
    T_step: float,
    freq: int,
    left_foot_id: int,
    right_foot_id: int,
    n_steps: int,
) -> Callable[
    [
        mjx.Model,
        mjx.Data,
        jnp.ndarray,
        jnp.ndarray,
        float,
        float,
        float,
    ],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    ticks_per_step = int(T_step * freq)
    g = 9.81

    @jax.jit
    def dead_beat_alip(
        mjx_model: mjx.Model,
        mjx_data: mjx.Data,
        com_height_des: jnp.ndarray,
        v_des: jnp.ndarray,
        gait_width: float,
        foot_height: float,
        floor_height: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        l = jnp.sqrt(g / com_height_des)
        m = mjx_model.body_subtreemass[mjx_model.body_rootid[0]]
        mH = m * com_height_des

        A_alip = jnp.array(
            [
                [0, 0, 0, 1 / mH],
                [0, 0, 1 / mH, 0],
                [0, m * g, 0, 0],
                [m * g, 0, 0, 0],
            ]
        )
        A_alip_lin = jnp.eye(4) + A_alip / freq

        @jax.jit
        def one_step_alip(carry, v_des):
            # Carry is:
            # com_world: CoM position in world frame (x, y, z, yaw)
            # L_init: CoM angular momentum
            # p_stance: stance foot position in world frame, (x, y, z, yaw)
            # p_swing: swing foot position in world frame (x, y, z, yaw)
            # step_sign: -1 if left foot is stance, 1 if right foot is stance
            com_init_world, L_init, p_stance_world, p_swing_world, step_sign = carry
            vx_des, vy_des, wz_des = v_des

            p_stance_yaw = p_stance_world[3]
            R_stance = jnp.array(
                [
                    [jnp.cos(p_stance_yaw), -jnp.sin(p_stance_yaw)],
                    [jnp.sin(p_stance_yaw), jnp.cos(p_stance_yaw)],
                ]
            )
            # Transform from world to local frames
            p_stance2com = R_stance.T @ (com_init_world[:2] - p_stance_world[:2])
            p_swing2com = R_stance.T @ (com_init_world[:2] - p_swing_world[:2])
            x0 = jnp.concatenate([p_stance2com, L_init])

            # Compute sinh and cosh once and for all
            sinh_lt, cosh_lt = jnp.sinh(l * T_step), jnp.cosh(l * T_step)

            # === Dead Beat ALIP
            # Computing desired next stance foot position
            # Formula 14
            # L_minus = mH * l * jnp.sinh(l * T_step) * p_stance2com[::-1] + jnp.cosh(l * T_step) * L_init
            L_minus = mH * l * sinh_lt * p_stance2com[::-1] + cosh_lt * L_init
            L_des = jnp.array(
                [
                    # Formula 16
                    step_sign * mH * gait_width * l * sinh_lt / (2 * (1 + cosh_lt)) + mH * vy_des,
                    # From linear matrix
                    mH * vx_des,
                ]
            )

            # Formula 15
            p_des = ((L_des - cosh_lt * L_minus) / (mH * l * sinh_lt))[::-1]

            # === Generating step trajectory
            s = jnp.linspace(0, 1, ticks_per_step)

            # == CoM trajectory
            # CoM trajectory relative to stance foot
            _, x_traj = jax.lax.scan(
                lambda x, _: (A_alip_lin @ x, A_alip_lin @ x),
                x0,
                None,
                length=ticks_per_step,
            )
            # Yaw trajectory
            _, com_yaw_traj = jax.lax.scan(
                lambda yaw, _: (yaw + wz_des / freq, yaw + wz_des / freq),
                com_init_world[3],
                None,
                length=ticks_per_step,
            )
            p_stance2com_traj, L_traj = x_traj[:, :2], x_traj[:, 2:]

            # World frame
            com_traj_world = jnp.hstack(
                [
                    (R_stance @ p_stance2com_traj.T).T + p_stance_world[:2],
                    jnp.ones((ticks_per_step, 1)) * com_height_des,
                    com_yaw_traj[:, None],
                ]
            )

            # == Swing foot trajectory
            # Relative to CoM
            swing_foot_traj = jnp.array(
                [
                    ((1 + jnp.cos(jnp.pi * s)) * p_swing2com[0] + (1 - jnp.cos(jnp.pi * s)) * p_des[0]) / 2,
                    ((1 + jnp.cos(jnp.pi * s)) * p_swing2com[1] + (1 - jnp.cos(jnp.pi * s)) * p_des[1]) / 2,
                    foot_height - 4 * foot_height * (s - 0.5) ** 2 + floor_height,
                ]
            ).T
            # Yaw trajectory
            _, swing_foot_yaw_traj = jax.lax.scan(
                lambda yaw, _: (yaw + 2 * wz_des / freq, yaw + 2 * wz_des / freq),
                p_swing_world[3],
                None,
                length=ticks_per_step,
            )
            swing_foot_traj_world = jnp.hstack(
                [
                    com_traj_world[:, :2] - (R_stance @ swing_foot_traj[:, :2].T).T,
                    swing_foot_traj[:, 2:3],
                    swing_foot_yaw_traj[:, None],
                ]
            )

            # == Stance foot trajectory
            stance_foot_traj_world = jnp.tile(p_stance_world, (ticks_per_step, 1))

            left_foot_traj = jax.lax.cond(
                step_sign == 1, lambda: swing_foot_traj_world, lambda: stance_foot_traj_world
            )
            right_foot_traj = jax.lax.cond(
                step_sign == 1, lambda: stance_foot_traj_world, lambda: swing_foot_traj_world
            )

            return (
                com_traj_world[-1],
                L_traj[-1],
                swing_foot_traj_world[-1],
                stance_foot_traj_world[-1],
                -step_sign,
            ), (
                com_traj_world,
                left_foot_traj,
                right_foot_traj,
            )

        body_root = mjx_model.body_rootid[0]

        # Initial parameters
        com0 = jnp.concatenate([mjx_data.subtree_com[body_root, :2], com_height_des.reshape(1), jnp.zeros(1)])
        L0 = mjx_data.subtree_angmom[body_root, :2]
        # First step -- with right leg!
        p_stance = jnp.concatenate([mjx_data.xpos[left_foot_id], jnp.zeros(1)])
        p_swing = jnp.concatenate([mjx_data.xpos[right_foot_id], jnp.zeros(1)])
        step_sign = -1

        # Scan the step function to get a rollout with sequence of steps
        _, (com_traj, left_foot_traj, right_foot_traj) = jax.lax.scan(
            one_step_alip,
            (com0, L0, p_stance, p_swing, step_sign),
            v_des,
        )

        return com_traj, left_foot_traj, right_foot_traj

    return dead_beat_alip


# === Mujoco ===
mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()

# --- Mujoco visualization ---
# Initialize render window and launch it at the background
vis = BatchVisualizer(MJCF_PATH, n_models=8, alpha=0.2, record=True)
com_marker_names = [f"com_marker_{i}" for i in range(vis.n_models)]
left_foot_marker_names = [f"left_foot_marker_{i}" for i in range(vis.n_models)]
right_foot_marker_names = [f"right_foot_marker_{i}" for i in range(vis.n_models)]

# === Mjinx ===
# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-1, v_max=1)

# Creating components of interest and adding them to the problem
joints_barrier = JointBarrier(
    "jnt_range",
    gain=0.1,
)
com_task = ComTask("com_task", cost=10.0, gain=50.0, mask=[1, 1, 1])
torso_task = FrameTask("torso_task", cost=1.0, gain=10.0, obj_name="cassie-pelvis", mask=[0, 0, 0, 1, 1, 1])

# Feet (in stance)
left_foot_task = FrameTask(
    "left_foot_task",
    cost=10.0,
    gain=50.0,
    obj_name="left-foot",
    mask=[1, 1, 1, 1, 0, 1],
)
right_foot_task = FrameTask(
    "right_foot_task",
    cost=10.0,
    gain=50.0,
    obj_name="right-foot",
    mask=[1, 1, 1, 1, 0, 1],
)

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
mjx_data = update(mjx_model, jnp.array(q0))
q = jnp.tile(q0, (N_batch, 1))
dt = 1e-2
ts = np.arange(0, 20, dt)
left_foot_id = mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_BODY, "left-foot")
right_foot_id = mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_BODY, "right-foot")

# === ALIP ===
T_step = 0.34
n_steps = int(ts[-1] / T_step)
# The angular velocity is selected so that the robot makes a full turn in the given time
v_des = jnp.concatenate(
    [
        jnp.array([0.3, 0.0, 0.0]) * jnp.ones((n_steps // 2, 3)),
        jnp.array([-0.3, 0.0, 0.0]) * jnp.ones((n_steps // 2, 3)),
    ]
)
floor_height = mjx_data.xpos[left_foot_id][2]

com0 = mjx_data.subtree_com[mjx_model.body_rootid[0]]
left_foot_quat = np.tile(
    np.array(mjx_data.xquat[left_foot_id]),
    (N_batch, 1),
)
right_foot_quat = np.tile(
    np.array(mjx_data.xquat[right_foot_id]),
    (N_batch, 1),
)
torso_quat = np.tile(
    np.array(mjx_data.xquat[mjx.name2id(mjx_model, mj.mjtObj.mjOBJ_BODY, "cassie-pelvis")]),
    (N_batch, 1),
)

foot_height = 0.03

dead_beat_fn = make_dead_beat_alip(
    T_step=T_step,
    freq=int(1 / dt),
    left_foot_id=left_foot_id,
    right_foot_id=right_foot_id,
    n_steps=n_steps,
)
com_traj, left_foot_traj, right_foot_traj = dead_beat_fn(
    mjx_model=mjx_model,
    mjx_data=mjx_data,
    v_des=v_des,
    com_height_des=com0[2] - floor_height,
    gait_width=mjx_data.xpos[right_foot_id][1] - mjx_data.xpos[left_foot_id][1],
    foot_height=foot_height,
    floor_height=floor_height,
)
com_traj = np.array(com_traj)
left_feet_traj = np.array(left_foot_traj)
right_feet_traj = np.array(right_foot_traj)

n_ticks = com_traj.shape[1]
offsets = np.floor(np.linspace(0, n_steps, num=N_batch, endpoint=False)).astype(int)

# Compiling the problem upon any parameters update
problem_data = problem.compile()
# --- Batching ---
print("Setting up batched computations...")
solver_data = jax.vmap(solver.init, in_axes=0)(v_init=jnp.zeros((N_batch, mjx_model.nv)))

with problem.set_vmap_dimension() as empty_problem_data:
    empty_problem_data.components["com_task"].target_com = 0
    empty_problem_data.components["torso_task"].target_frame = 0
    empty_problem_data.components["left_foot_task"].target_frame = 0
    empty_problem_data.components["right_foot_task"].target_frame = 0

# Vmapping solve and integrate functions.
solve_jit = jax.jit(
    jax.vmap(
        solver.solve,
        in_axes=(0, 0, empty_problem_data),
    )
)
integrate_jit = jax.jit(jax.vmap(integrate, in_axes=(None, 0, 0, None)), static_argnames=["dt"])

# === Control loop ===


def quat_mult(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions q and r.
    Both q and r are assumed to have shape (..., 4) with scalar-first convention.
    """
    w = q[..., 0] * r[..., 0] - q[..., 1] * r[..., 1] - q[..., 2] * r[..., 2] - q[..., 3] * r[..., 3]
    x = q[..., 0] * r[..., 1] + q[..., 1] * r[..., 0] + q[..., 2] * r[..., 3] - q[..., 3] * r[..., 2]
    y = q[..., 0] * r[..., 2] - q[..., 1] * r[..., 3] + q[..., 2] * r[..., 0] + q[..., 3] * r[..., 1]
    z = q[..., 0] * r[..., 3] + q[..., 1] * r[..., 2] - q[..., 2] * r[..., 1] + q[..., 3] * r[..., 0]
    return np.stack([w, x, y, z], axis=-1)


# Convert a yaw angle to a quaternion (scalar-first) and compose with an initial orientation.
def yaw_to_quat(yaw: np.ndarray, init_quat: np.ndarray) -> np.ndarray:
    # Compute the additional yaw rotation quaternion: rotation about z-axis.
    q_yaw = np.stack(
        [
            np.cos(yaw / 2),
            np.zeros_like(yaw),
            np.zeros_like(yaw),
            np.sin(yaw / 2),
        ],
        axis=-1,
    )
    # Compose the initial orientation with the yaw rotation.
    return quat_mult(
        q_yaw,
        init_quat,
    )


try:
    for i in range(len(ts)):
        t = ts[i]
        # Compute the current tick based on time
        current_tick = int(i % n_ticks)
        # Compute the current step based on the time
        current_step = i // n_ticks
        # Each robot gets a different tick index by adding its offset
        step_indices = (current_step + offsets) % n_steps

        # Extract the (xyz, yaw) values for each task.
        com_vals = com_traj[step_indices, current_tick, :]  # shape: (N_batch, 4)
        left_vals = left_feet_traj[step_indices, current_tick, :]  # shape: (N_batch, 4)
        right_vals = right_feet_traj[step_indices, current_tick, :]  # shape: (N_batch, 4)

        # Convert yaw to quaternion for each task
        torso_quat = yaw_to_quat(com_vals[:, 3], torso_quat)
        left_quat = yaw_to_quat(left_vals[:, 3], left_foot_quat)
        right_quat = yaw_to_quat(right_vals[:, 3], right_foot_quat)

        # Form the SE3 target vectors for the tasks (concatenate xyz with quaternion)
        left_target_se3 = np.concatenate([left_vals[:, :3], left_quat], axis=1)
        right_target_se3 = np.concatenate([right_vals[:, :3], right_quat], axis=1)

        # Assign these SE3 targets to the tasks.
        com_task.target_com = com_vals[:, :3]
        torso_task.target_frame = np.concatenate([np.zeros((N_batch, 3)), torso_quat], axis=1)
        left_foot_task.target_frame = left_target_se3
        right_foot_task.target_frame = right_target_se3

        problem_data = problem.compile()
        opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
        # Integrating
        q = integrate_jit(
            mjx_model,
            q,
            opt_solution.v_opt,
            dt,
        )
        # --- MuJoCo visualization ---
        indices = np.arange(0, N_batch, N_batch // vis.n_models)

        vis.update(q[:: N_batch // vis.n_models])

except KeyboardInterrupt:
    print("Finalizing the simulation as requested...")
except Exception:
    print(traceback.format_exc())
finally:
    if vis.record:
        vis.save_video(round(1 / dt))
    vis.close()
