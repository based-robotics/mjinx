from pathlib import Path
from time import perf_counter

import mujoco
import mujoco.viewer
import numpy as np
from robot_descriptions.panda_mj_description import MJCF_PATH

import mink

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    ## =================== ##
    ## Setup IK.
    ## =================== ##

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="hand",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model=model, cost=1e-2),
    ]

    ## =================== ##

    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    dt = 1e-2
    max_iters = int(20 / dt)

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)
    mujoco.mj_forward(model, data)

    solve_times = []
    integrate_times = []
    n_steps = 0

    try:
        # Compute velocity and integrate into the next configuration.
        for i in range(max_iters):
            # Update task target.
            T_wt = mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(np.eye(3)), np.array([0.2, 0.4, 0.2 + 0.2 * np.sin(i * dt) ** 2])
            )
            end_effector_task.set_target(T_wt)
            t1 = perf_counter()
            configuration.update(data.qpos)
            vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
            t2 = perf_counter()
            solve_times.append(t2 - t1)

            t1 = perf_counter()
            configuration.integrate_inplace(vel, dt)
            t2 = perf_counter()
            integrate_times.append(t2 - t1)

            data.qpos = configuration.q
            mujoco.mj_fwdPosition(model, data)

            n_steps += 1
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
