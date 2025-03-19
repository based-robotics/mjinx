from pathlib import Path
from time import perf_counter

import numpy as np
import mujoco
import mink
import os

if __name__ == "__main__":
    WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH = os.path.join(WORKSPACE_ROOT, "examples/g1_description/g1.xml")
    # Load model and create data.
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)

    # Set up configuration and tasks.
    configuration = mink.Configuration(model)
    configuration.update(data.qpos)

    # Define tasks.
    pelvis_orientation_task = mink.FrameTask(
        frame_name="pelvis",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    torso_orientation_task = mink.FrameTask(
        frame_name="torso_link",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model, cost=1e-1)
    com_task = mink.ComTask(cost=10.0)

    # Feet and hand tasks.
    feet = ["right_foot", "left_foot"]
    hands = ["right_wrist", "left_wrist"]

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)

    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=5.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)

    tasks = [
        pelvis_orientation_task,
        torso_orientation_task,
        posture_task,
        com_task,
    ]
    tasks.extend(feet_tasks)
    tasks.extend(hand_tasks)

    # Create collision pairs from all collision bodies (pairwise checks except for excluded collisions).
    collision_pairs = [
        # Pairs with torso_geom.
        (
            ["torso_geom"],
            ["left_shoulder_geom", "left_elbow_geom", "right_shoulder_geom", "right_elbow_geom"],
        ),
        # Pairs between left and right shoulder/elbow geoms (excluding the specified excluded pairs).
        (["left_shoulder_geom"], ["right_shoulder_geom", "right_elbow_geom"]),
        (["left_elbow_geom"], ["right_shoulder_geom", "right_elbow_geom"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )
    limits = [mink.ConfigurationLimit(model), collision_avoidance_limit]

    # Set initial targets for pelvis and torso based on configuration.
    pelvis_orientation_task.set_target_from_configuration(configuration)
    torso_orientation_task.set_target_from_configuration(configuration)
    posture_task.set_target_from_configuration(configuration)

    # For defined trajectories we use initial COM and fixed offsets.
    # Get initial center-of-mass.
    mujoco.mj_forward(model, data)
    com0 = data.subtree_com[1].copy()

    # Simulation parameters.
    solver = "quadprog"
    dt = 1e-2
    total_time = 20.0
    n_steps = 0

    solve_times = []
    integrate_times = []

    print("Starting simulation with defined trajectory...")
    t0 = perf_counter()
    t_sim = 0.0

    # Main simulation loop.
    while t_sim < total_time:
        # -- Update target trajectories --
        # COM target: modulate height with a sine.
        com_target = com0 + np.array([0.0, 0.0, 0.1 * np.sin(t_sim)])
        com_task.set_target(com_target)

        # Hand trajectories: keep a constant target with a slight oscillation.
        for i, hand_task in enumerate(hand_tasks):
            phase = np.pi / 2 if i == 0 else 0.0
            pos = np.array([-0.1, 0.25 * (1 if i == 0 else -1), 0.8 + 0.03 * np.sin(t_sim + phase)])
            T_target = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(np.eye(3)), pos)
            hand_task.set_target(T_target)

        # -- Solve the IK --
        t1 = perf_counter()
        vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-1, limits=limits)
        t2 = perf_counter()
        solve_times.append(t2 - t1)

        # -- Integrate the configuration --
        t1 = perf_counter()
        configuration.integrate_inplace(vel, dt)
        t2 = perf_counter()
        integrate_times.append(t2 - t1)

        # Update the data.qpos from the configuration.
        data.qpos = configuration.q
        mujoco.mj_fwdPosition(model, data)

        n_steps += 1
        t_sim += dt

    t_total = perf_counter() - t0

    # -- Performance Report --
    print("\n=== Performance Report ===")
    print(f"Total steps completed: {n_steps}")
    if solve_times:
        avg_solve = sum(solve_times) / len(solve_times)
        std_solve = np.std(solve_times)
        print(f"Solve:    {avg_solve * 1000:8.3f} ± {std_solve * 1000:8.3f} ms")
    if integrate_times:
        avg_integrate = sum(integrate_times) / len(integrate_times)
        std_integrate = np.std(integrate_times)
        print(f"Integrate:{avg_integrate * 1000:8.3f} ± {std_integrate * 1000:8.3f} ms")
    if solve_times and integrate_times:
        avg_total = sum(s + i for s, i in zip(solve_times, integrate_times)) / len(solve_times)
        print(f"\nAverage computation time per step: {avg_total * 1000:.3f} ms")
        print(f"Effective computation rate: {1 / avg_total:.1f} Hz")
    print(f"\nTotal simulation duration: {t_total:.3f} seconds")
