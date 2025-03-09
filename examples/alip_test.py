from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # necessary import for 3D plotting
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
            [0, 0, -1 / mH, 0],
            [0, -m * g, 0, 0],
            [m * g, 0, 0, 0],
        ]
    )
    B_alip = jnp.array(
        [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
        ]
    ).T

    A_alip_lin = jnp.eye(4) + A_alip / freq
    B_alip_lin = B_alip / freq

    @jax.jit
    def rollout_step(carry, v_des):
        # Carry is:
        # com_world: CoM position in world frame
        # L_init: CoM angular momentum
        # p_stance: stance foot position in world frame
        # p_swing: swing foot position in world frame
        # step_sign: -1 if left foot is stance, 1 if right foot is stance
        com_init, L_init, p_stance, p_swing, step_sign = carry
        vx_des, vy_des, wz_des = v_des

        # Transform from world to local frames
        p_stance2com = com_init - p_stance[:2]
        p_swing2com = com_init - p_swing[:2]
        x0 = jnp.concatenate([p_stance2com, L_init])

        jax.debug.print("p_stance2com={a} p_swing2com={b}", a=p_stance2com, b=p_swing2com)

        # Computing desired next stance foot position
        # Formula 14
        L_minus = mH * l * jnp.sinh(l * T_step) * p_stance2com[::-1] + jnp.cosh(l * T_step) * L_init
        jax.debug.print("first_term_x={b} L_minus={a}", b=mH * l * jnp.sinh(l * T_step), a=L_minus)
        L_des = jnp.array(
            [
                # Formula 16
                -step_sign * mH * gait_width * l * jnp.sinh(l * T_step) / (1 + jnp.cosh(l * T_step)) + mH * vy_des,
                # From linear matrix
                mH * vx_des,
            ]
        )
        jax.debug.print("L_des={a}", a=L_des)

        # Formula 15
        p_des = ((L_des - jnp.cosh(l * T_step) * L_minus) / (mH * l * jnp.sinh(l * T_step)))[::-1]
        jax.debug.print("p_des={a}", a=p_des)

        # === Generating step trajectory
        s = jnp.linspace(0, 1, ticks_per_step)

        # == CoM trajectory
        # CoM trajectory relative to stance foot
        _, x_traj_history = jax.lax.scan(
            lambda x, _: (A_alip_lin @ x, A_alip_lin @ x),
            x0,
            None,
            length=ticks_per_step - 1,
        )
        x_traj = jnp.vstack([x0, x_traj_history])
        p_stance2com_traj, L_traj = x_traj[:, :2], x_traj[:, 2:]
        jax.debug.print("x0={b}, x_traj={a}", b=x0, a=x_traj[-1])

        # World frame
        com_traj_world = p_stance2com_traj + p_stance[:2]

        # == Swing foot trajectory
        # Relative to CoM
        swing_foot_traj = jnp.array(
            [
                ((1 + jnp.cos(jnp.pi * s)) * p_swing2com[0] + (1 - jnp.cos(jnp.pi * s)) * p_des[0]) / 2,
                ((1 + jnp.cos(jnp.pi * s)) * p_swing2com[1] + (1 - jnp.cos(jnp.pi * s)) * p_des[1]) / 2,
                4 * foot_height * (s - 0.5) ** 2 - foot_height,
            ]
        ).T
        jax.debug.print(
            "swing_foot_traj_init={b} swing_foot_traj={a}",
            b=swing_foot_traj[0],
            a=swing_foot_traj[-1],
        )
        swing_foot_traj_world = jnp.hstack(
            [
                (com_traj_world - swing_foot_traj[:, :2]),
                (com_height_des - swing_foot_traj[:, 2]).reshape(-1, 1),
            ]
        )
        jax.debug.print(
            "swing_foot_init={b}, swing_foot_traj_world={a}",
            b=swing_foot_traj_world[0],
            a=swing_foot_traj_world[-1],
        )

        # == Stance foot trajectory
        stance_foot_traj = jnp.tile(p_stance, (ticks_per_step, 1))

        left_foot_traj = jax.lax.cond(step_sign == 1, lambda: swing_foot_traj_world, lambda: stance_foot_traj)
        right_foot_traj = jax.lax.cond(step_sign == 1, lambda: stance_foot_traj, lambda: swing_foot_traj_world)
        jax.debug.print("-" * 90)
        return (
            com_traj_world[-1],
            L_traj[-1],
            swing_foot_traj_world[-1],
            stance_foot_traj[-1],
            -step_sign,
        ), (
            com_traj_world,
            left_foot_traj,
            right_foot_traj,
        )

    mjx_data: mjx.Data = mjx.make_data(model).replace(qpos=q)
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

    return np.array(x_traj[:2]), np.array(left_foot_traj), np.array(right_foot_traj)


def visualize_motion(left_leg: np.ndarray, right_leg: np.ndarray, com: np.ndarray):
    """
    Visualizes the 3D motion paths for the left leg, right leg, and center of mass (CoM).

    Args:
        left_leg (np.ndarray): Array of shape (N, 3) with left leg positions.
        right_leg (np.ndarray): Array of shape (N, 3) with right leg positions.
        com (np.ndarray): Array of shape (N, 3) with CoM positions.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        left_leg[:, 0],
        left_leg[:, 1],
        left_leg[:, 2],
        label="Left Leg",
        color="red",
        lw=2,
    )
    ax.plot(
        right_leg[:, 0],
        right_leg[:, 1],
        right_leg[:, 2],
        label="Right Leg",
        color="blue",
        lw=2,
    )
    ax.plot(com[:, 0], com[:, 1], com[:, 2], label="Center of Mass", color="green", lw=2)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.set_title("3D Motion Trajectories")

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
    com0 = data.subtree_com[model.body_rootid[0]]
    left_foot_id = mjx.name2id(model, mj.mjtObj.mjOBJ_BODY, "left-foot")
    right_foot_id = mjx.name2id(model, mj.mjtObj.mjOBJ_BODY, "right-foot")

    q = data.qpos
    com_traj, left_foot_traj, right_foot_traj = dead_beat_alip(
        model=model,
        q=q,
        v_des=jnp.array([0.0, 0.0, 0.0]),
        com_height_des=com0[2],
        T_step=0.3,
        freq=100,
        gait_width=data.xpos[right_foot_id][1] - data.xpos[left_foot_id][1],
        foot_height=0.03,
        left_foot_id=left_foot_id,
        right_foot_id=right_foot_id,
        n_steps=3,
    )

    visualize_motion(left_foot_traj, right_foot_traj, com_traj)
