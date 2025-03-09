from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # necessary import for 3D plotting
from robot_descriptions.cassie_mj_description import MJCF_PATH
import datetime
from matplotlib.animation import FuncAnimation, PillowWriter


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
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    @jax.jit
    def rollout_step(carry, v_des):
        # Carry is:
        # com_world: CoM position in world frame
        # L_init: CoM angular momentum
        # p_stance: stance foot position in world frame
        # p_swing: swing foot position in world frame
        # step_sign: -1 if left foot is stance, 1 if right foot is stance
        com_init_world, L_init, p_stance_world, p_swing_world, step_sign = carry
        vx_des, vy_des, wz_des = v_des

        # Transform from world to local frames
        p_stance2com = com_init_world - p_stance_world[:2]
        p_swing2com = com_init_world - p_swing_world[:2]
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
        jax.debug.print("L_minus: {x}", x=L_minus)
        jax.debug.print("L_des: {x}", x=L_des)
        jax.debug.print("-" * 40)

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
        # x_traj = jnp.vstack([x0, x_traj_history])
        p_stance2com_traj, L_traj = x_traj[:, :2], x_traj[:, 2:]

        # World frame
        com_traj_world = p_stance2com_traj + p_stance_world[:2]

        # == Swing foot trajectory
        # Relative to CoM
        swing_foot_traj = jnp.array(
            [
                ((1 + jnp.cos(jnp.pi * s)) * p_swing2com[0] + (1 - jnp.cos(jnp.pi * s)) * p_des[0]) / 2,
                ((1 + jnp.cos(jnp.pi * s)) * p_swing2com[1] + (1 - jnp.cos(jnp.pi * s)) * p_des[1]) / 2,
                foot_height - 4 * foot_height * (s - 0.5) ** 2 + floor_height,
            ]
        ).T
        swing_foot_traj_world = jnp.hstack(
            [
                (com_traj_world - swing_foot_traj[:, :2]),
                swing_foot_traj[:, 2:3],
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
    com0 = mjx_data.subtree_com[body_root, :2]
    L0 = mjx_data.subtree_angmom[body_root, :2]
    print("L0: ", L0)
    # First step -- with right leg!
    p_stance = mjx_data.xpos[left_foot_id]
    p_swing = mjx_data.xpos[right_foot_id]
    step_sign = -1

    _, (x_traj, left_foot_traj, right_foot_traj) = jax.lax.scan(
        rollout_step,
        (com0, L0, p_stance, p_swing, step_sign),
        v_des * jnp.ones((n_steps, 3)),
    )
    com_traj = np.concatenate(
        [
            np.array(x_traj[:, :, :2]),
            np.ones((n_steps, ticks_per_step, 1)) * com_height_des,
        ],
        axis=2,
    )

    return com_traj, np.array(left_foot_traj), np.array(right_foot_traj)


def visualize_motion(com: np.ndarray, left_leg: np.ndarray, right_leg: np.ndarray, save_gif: bool = False):
    """
    Creates a 3D animation of CoM, left foot, and right foot trajectories.

    Args:
        com (np.ndarray): Array of shape (num_steps, 3, ticks_per_step) for CoM positions.
        left_leg (np.ndarray): Array of shape (num_steps, 3, ticks_per_step) for left foot positions.
        right_leg (np.ndarray): Array of shape (num_steps, 3, ticks_per_step) for right foot positions.
        save_gif (bool): Optionally save the animation as a GIF with a timestamp.
    """
    num_steps, ticks, _ = com.shape
    total_frames = num_steps * ticks

    # Compute global min/max for setting axis limits
    all_points = np.concatenate([com.reshape(-1, 3), left_leg.reshape(-1, 3), right_leg.reshape(-1, 3)], axis=0)
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
    ax.set_title("3D Motion Animation")

    # Create markers for the three points
    (com_point,) = ax.plot([], [], [], "go", markersize=8, label="Center of Mass")
    (left_point,) = ax.plot([], [], [], "ro", markersize=8, label="Left Foot")
    (right_point,) = ax.plot([], [], [], "bo", markersize=8, label="Right Foot")
    ax.legend()
    step_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def init():
        com_point.set_data([], [])
        com_point.set_3d_properties([])
        left_point.set_data([], [])
        left_point.set_3d_properties([])
        right_point.set_data([], [])
        right_point.set_3d_properties([])
        step_text.set_text("")
        return com_point, left_point, right_point, step_text

    def update(frame):
        step_idx = frame // ticks
        tick_idx = frame % ticks
        cp = com[step_idx, tick_idx, :]
        lp = left_leg[step_idx, tick_idx, :]
        rp = right_leg[step_idx, tick_idx, :]

        com_point.set_data([cp[0]], [cp[1]])
        com_point.set_3d_properties([cp[2]])
        left_point.set_data([lp[0]], [lp[1]])
        left_point.set_3d_properties([lp[2]])
        right_point.set_data([rp[0]], [rp[1]])
        right_point.set_3d_properties([rp[2]])
        step_text.set_text(f"Step: {step_idx + 1}")
        return com_point, left_point, right_point, step_text

    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True, interval=100)

    if save_gif:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"motion_animation_{timestamp}.gif"
        writer = PillowWriter(fps=10)
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
        v_des=jnp.array([0.4, 1.0, 0.0]),
        com_height_des=com0[2] - floor_height,
        T_step=0.2,
        freq=100,
        gait_width=data.xpos[right_foot_id][1] - data.xpos[left_foot_id][1],
        foot_height=0.03,
        floor_height=floor_height,
        left_foot_id=left_foot_id,
        right_foot_id=right_foot_id,
        n_steps=10,
    )
    visualize_motion(com_traj, left_foot_traj, right_foot_traj)
