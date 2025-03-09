import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from matplotlib.animation import FFMpegFileWriter, FuncAnimation
from robot_descriptions.cassie_mj_description import MJCF_PATH


def dead_beat_alip(
    model: mjx.Model,
    q: jnp.ndarray,
    v_des: jnp.ndarray,
    com_height_des: jnp.ndarray,
    T_step: float,
    freq: int,
    gait_width: float,
    foot_height: float,
    floor_height: float,
    left_foot_id: float,
    right_foot_id: float,
    n_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    ticks_per_step = int(T_step * freq)
    g = 9.81
    l = jnp.sqrt(g / com_height_des)
    m = model.body_subtreemass[model.body_rootid[0]]
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

    def yaw_to_R(yaw: jnp.ndarray) -> jnp.ndarray:
        return jnp.array(
            [
                [jnp.cos(yaw), -jnp.sin(yaw)],
                [jnp.sin(yaw), jnp.cos(yaw)],
            ]
        )

    @jax.jit
    def rollout_step(carry, v_des):
        # Carry is:
        # com_world: CoM position in world frame (x, y, yaw)
        # L_init: CoM angular momentum
        # p_stance: stance foot position in world frame, (x, y, z, yaw)
        # p_swing: swing foot position in world frame (x, y, z, yaw)
        # step_sign: -1 if left foot is stance, 1 if right foot is stance
        com_init_world, L_init, p_stance_world, p_swing_world, step_sign = carry
        vx_des, vy_des, wz_des = v_des

        p_stance_yaw = p_stance[2]
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

        # Computing desired next stance foot position
        # Formula 14
        # L_minus = mH * l * jnp.sinh(l * T_step) * p_stance2com[::-1] + jnp.cosh(l * T_step) * L_init
        sinh_lt, cosh_lt = jnp.sinh(l * T_step), jnp.cosh(l * T_step)
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
            com_init_world[2],
            None,
            length=ticks_per_step,
        )
        p_stance2com_traj, L_traj = x_traj[:, :2], x_traj[:, 2:]

        # World frame
        com_traj_world = jnp.hstack(
            [
                (R_stance @ p_stance2com_traj.T).T + p_stance_world[:2],
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

        left_foot_traj = jax.lax.cond(step_sign == 1, lambda: swing_foot_traj_world, lambda: stance_foot_traj_world)
        right_foot_traj = jax.lax.cond(step_sign == 1, lambda: stance_foot_traj_world, lambda: swing_foot_traj_world)
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

    mjx_data: mjx.Data = mjx.make_data(model).replace(qpos=jnp.array(q0))
    mjx_data = mjx.fwd_position(model, mjx_data)
    body_root = model.body_rootid[0]

    com0 = jnp.concatenate([mjx_data.subtree_com[body_root, :2], jnp.zeros(1)])
    L0 = mjx_data.subtree_angmom[body_root, :2]
    # First step -- with right leg!
    p_stance = jnp.concatenate([mjx_data.xpos[left_foot_id], jnp.zeros(1)])
    p_swing = jnp.concatenate([mjx_data.xpos[right_foot_id], jnp.zeros(1)])
    step_sign = -1

    _, (x_traj, left_foot_traj, right_foot_traj) = jax.lax.scan(
        rollout_step,
        (com0, L0, p_stance, p_swing, step_sign),
        v_des * jnp.ones((n_steps, 3)),
    )
    # Adding constant height of CoM to the trajectory
    com_traj = np.concatenate(
        [
            np.array(x_traj[:, :, :2]),
            np.ones((n_steps, ticks_per_step, 1)) * com_height_des,
            np.array(x_traj[:, :, 2:3]),
        ],
        axis=2,
    )

    return com_traj, np.array(left_foot_traj), np.array(right_foot_traj)


def visualize_motion(
    com: np.ndarray, left_leg: np.ndarray, right_leg: np.ndarray, save: bool = False, freq: int = 100
):
    """
    Creates a 3D animation of CoM (torso) and feet trajectories with orientation.
    All input arrays are expected to be of shape (num_steps, ticks, 4) where the last dimension
    represents (x, y, z, yaw). Each foot and the torso are drawn as a short segment indicating their yaw orientation,
    and a dashed line connects the torso's base to the midpoint of each foot (average of foot base and orientation endpoint).

    Args:
        com (np.ndarray): Array of shape (num_steps, 4, ticks) for torso positions and orientation.
        left_leg (np.ndarray): Array of shape (num_steps, 4, ticks) for left foot (x, y, z, yaw).
        right_leg (np.ndarray): Array of shape (num_steps, 4, ticks) for right foot (x, y, z, yaw).
        save (bool): Optionally save the animation as a video with a timestamp.
        freq (int): Frequency (fps) for the animation.
    """
    num_steps, ticks, _ = com.shape[:3]
    total_frames = num_steps * ticks
    foot_orient_length = 0.05
    torso_orient_length = 0.05

    # Compute endpoints for orientation segments to adjust axis limits
    left_endpoints = left_leg[..., :3] + foot_orient_length * np.stack(
        [np.cos(left_leg[..., 3]), np.sin(left_leg[..., 3]), np.zeros_like(left_leg[..., 3])], axis=-1
    )
    right_endpoints = right_leg[..., :3] + foot_orient_length * np.stack(
        [np.cos(right_leg[..., 3]), np.sin(right_leg[..., 3]), np.zeros_like(right_leg[..., 3])], axis=-1
    )
    torso_endpoints = com[..., :3] + torso_orient_length * np.stack(
        [np.cos(com[..., 3]), np.sin(com[..., 3]), np.zeros_like(com[..., 3])], axis=-1
    )

    all_points = np.concatenate(
        [
            com[..., :3].reshape(-1, 3),
            left_leg[..., :3].reshape(-1, 3),
            right_leg[..., :3].reshape(-1, 3),
            left_endpoints.reshape(-1, 3),
            right_endpoints.reshape(-1, 3),
            torso_endpoints.reshape(-1, 3),
        ],
        axis=0,
    )
    x_min, y_min, z_min = all_points.min(axis=0) - 0.1
    x_max, y_max, z_max = all_points.max(axis=0) + 0.1

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Create the plot objects
    (com_point,) = ax.plot([], [], [], "go", markersize=8, label="CoM")
    (torso_line,) = ax.plot([], [], [], "k-", lw=2, label="Torso Orientation")
    (left_foot_line,) = ax.plot([], [], [], "r-", lw=2, label="Left Foot Orientation")
    (right_foot_line,) = ax.plot([], [], [], "b-", lw=2, label="Right Foot Orientation")
    (connection_line_left,) = ax.plot([], [], [], "r--", lw=1)
    (connection_line_right,) = ax.plot([], [], [], "b--", lw=1)
    ax.legend()
    step_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def init():
        com_point.set_data([], [])
        com_point.set_3d_properties([])
        torso_line.set_data([], [])
        torso_line.set_3d_properties([])
        left_foot_line.set_data([], [])
        left_foot_line.set_3d_properties([])
        right_foot_line.set_data([], [])
        right_foot_line.set_3d_properties([])
        connection_line_left.set_data([], [])
        connection_line_left.set_3d_properties([])
        connection_line_right.set_data([], [])
        connection_line_right.set_3d_properties([])
        step_text.set_text("")
        return (
            com_point,
            torso_line,
            left_foot_line,
            right_foot_line,
            connection_line_left,
            connection_line_right,
            step_text,
        )

    def update(frame):
        step_idx = frame // ticks
        tick_idx = frame % ticks

        cp = com[step_idx, tick_idx, :4]
        left = left_leg[step_idx, tick_idx, :4]
        right = right_leg[step_idx, tick_idx, :4]

        # Compute orientation endpoints using yaw
        torso_endpoint = np.array(
            [
                cp[0] + torso_orient_length * np.cos(cp[3]),
                cp[1] + torso_orient_length * np.sin(cp[3]),
                cp[2],
            ]
        )
        left_endpoint = np.array(
            [
                left[0] + foot_orient_length * np.cos(left[3]),
                left[1] + foot_orient_length * np.sin(left[3]),
                left[2],
            ]
        )
        right_endpoint = np.array(
            [
                right[0] + foot_orient_length * np.cos(right[3]),
                right[1] + foot_orient_length * np.sin(right[3]),
                right[2],
            ]
        )

        # Foot center as average of foot base and orientation endpoint
        left_center = (left[:3] + left_endpoint) / 2
        right_center = (right[:3] + right_endpoint) / 2

        # Update torso base marker and orientation line
        com_point.set_data([cp[0]], [cp[1]])
        com_point.set_3d_properties([cp[2]])
        torso_line.set_data([cp[0], torso_endpoint[0]], [cp[1], torso_endpoint[1]])
        torso_line.set_3d_properties([cp[2], torso_endpoint[2]])

        # Update foot orientation lines
        left_foot_line.set_data([left[0], left_endpoint[0]], [left[1], left_endpoint[1]])
        left_foot_line.set_3d_properties([left[2], left_endpoint[2]])

        right_foot_line.set_data([right[0], right_endpoint[0]], [right[1], right_endpoint[1]])
        right_foot_line.set_3d_properties([right[2], right_endpoint[2]])

        # Update connection lines from torso base to foot centers
        connection_line_left.set_data([cp[0], left_center[0]], [cp[1], left_center[1]])
        connection_line_left.set_3d_properties([cp[2], left_center[2]])

        connection_line_right.set_data([cp[0], right_center[0]], [cp[1], right_center[1]])
        connection_line_right.set_3d_properties([cp[2], right_center[2]])

        step_text.set_text(f"Step: {step_idx + 1}")
        return (
            com_point,
            torso_line,
            left_foot_line,
            right_foot_line,
            connection_line_left,
            connection_line_right,
            step_text,
        )

    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init, interval=1000 // freq)

    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"motion_animation_{timestamp}.mp4"
        writer = FFMpegFileWriter(fps=freq)
        anim.save(filename, writer=writer)
        print(f"Animation saved as {filename}")

    plt.show()


if __name__ == "__main__":
    # Example: generate sample data
    N = 200
    t = np.linspace(0, 4 * np.pi, N)

    # Sample trajectories -- replace these with your actual data.
    model = mj.MjModel.from_xml_path(MJCF_PATH)
    q0 = model.keyframe("home").qpos
    model = mjx.put_model(model)
    data = mjx.make_data(model).replace(qpos=jnp.array(q0))
    data = mjx.fwd_position(model, data)
    left_foot_id = mjx.name2id(model, mj.mjtObj.mjOBJ_BODY, "left-foot")
    right_foot_id = mjx.name2id(model, mj.mjtObj.mjOBJ_BODY, "right-foot")
    lf = data.xpos[left_foot_id]
    rf = data.xpos[right_foot_id]
    com0 = data.subtree_com[model.body_rootid[0]]
    floor_height = data.xpos[left_foot_id][2]

    com_traj, left_foot_traj, right_foot_traj = dead_beat_alip(
        model=model,
        q=q0,
        v_des=jnp.array([0.2, 0.0, 0.5]),
        com_height_des=com0[2] - floor_height,
        T_step=0.2,
        freq=100,
        gait_width=data.xpos[right_foot_id][1] - data.xpos[left_foot_id][1],
        foot_height=0.03,
        floor_height=floor_height,
        left_foot_id=left_foot_id,
        right_foot_id=right_foot_id,
        n_steps=20,
    )
    visualize_motion(com_traj, left_foot_traj, right_foot_traj, freq=100)
