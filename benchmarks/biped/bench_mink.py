"""IK with two four-bar linkages on Agility Cassie."""

from pathlib import Path
from time import perf_counter

import mink
import mujoco
import mujoco.viewer
import numpy as np
from mink.lie import SE3
from robot_descriptions.cassie_mj_description import MJCF_PATH


def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    configuration = mink.Configuration(model)

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="cassie-pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-2),
        com_task := mink.ComTask(cost=200.0),
    ]

    # Note: By not providing `equality_name`, all equality constraints in the model
    # will be regulated.
    equality_task = mink.EqualityConstraintTask(
        model=model,
        cost=500.0,
        gain=0.5,
        lm_damping=1e-3,
    )
    tasks.append(equality_task)

    feet = ["left-foot", "right-foot"]
    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="body",  # Cassie uses body for feet, not sites.
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    model = configuration.model
    data = configuration.data
    solver = "quadprog"
    dt = 1e-2
    max_iters = int(20 / dt)

    configuration.update_from_keyframe("home")
    posture_task.set_target_from_configuration(configuration)
    pelvis_orientation_task.set_target_from_configuration(configuration)
    feet_tasks[0].set_target_from_configuration(configuration)
    feet_tasks[1].set_target_from_configuration(configuration)

    com0 = configuration.data.subtree_com[model.body_rootid[0]]

    solve_times = []
    integrate_times = []
    n_steps = 0
    try:
        for i in range(max_iters):
            com_task.set_target([0.0, 0.0, com0[2] - 0.3 * np.sin(i / max_iters + np.pi / 2) ** 2])

            t1 = perf_counter()
            vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-1)
            t2 = perf_counter()
            solve_times.append(t2 - t1)

            t1 = perf_counter()
            configuration.integrate_inplace(vel, dt)
            t2 = perf_counter()
            integrate_times.append(t2 - t1)

            mujoco.mj_camlight(model, data)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
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
            avg_total = sum(t1 + t2 for t1, t2 in zip(solve_times, integrate_times)) / len(solve_times)
            print(f"\nAverage computation time per step: {avg_total * 1000:.3f} ms")
            print(f"Effective computation rate: {1 / avg_total:.1f} Hz")


if __name__ == "__main__":
    main()
