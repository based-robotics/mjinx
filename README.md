# MJINX
<!-- [![mypy](https://img.shields.io/github/actions/workflow/status/based-robotics/mjinx/mypy.yaml?branch=main)](https://github.com/based-robotics/mjinx/actions)
[![ruff](https://img.shields.io/github/actions/workflow/status/based-robotics/mjinx/ruff.yaml?branch=main)](https://github.com/based-robotics/mjinx/actions)
[![build](https://img.shields.io/github/actions/workflow/status/based-robotics/mjinx/build.yaml?branch=main)](https://github.com/based-robotics/mjinx/actions)
[![PyPI version](https://img.shields.io/pypi/v/jaxadi?color=blue)](https://pypi.org/project/jaxadi/)
[![PyPI downloads](https://img.shields.io/pypi/dm/jaxadi?color=blue)](https://pypistats.org/packages/jaxadi) -->


**Mjinx** is a library for auto-differentiable numerical inverse kinematics, powered by **JAX** and **Mujoco MJX**. The library was heavily inspired by the similar Pinocchio-based tool [pink](https://github.com/stephane-caron/pink/tree/main) and Mujoco-based analogue [mink](https://github.com/kevinzakka/mink/tree/main).

<!-- <div align="center">
  <img src="img/local_ik_output.gif" style="width: 45%; max-width: 300px" />
  <img src="img/go2_stance.gif" style="width: 45%; max-width: 300px" /> 
</div> -->
<p align="center">
  <img src="img/local_ik_output.gif" style="width: 300px" />
  <img src="img/go2_stance.gif" style="width: 300px" /> 
  <img src="img/local_ik_input.gif" style="width: 300px"/>
</p>

## Key features
1. *Flexibility*. Each control problem is assembled via `Components`, which enforce desired behaviour or keeps system in a safety set. 
2. *Different solution approaches*. `JAX` (i.e. it's efficient sampling and autodifferentiation) allows to implement variety of solvers, which might be more beneficial in different scenarios.
3. *Fully Jax-compatible*. Both optimal control problem and its solver are jax-compatible: jit-compilation and automatic vectorization are available for the whole problem.
4. *Convinience*. The functionality is nicely wrapped to make the interaction with it easier.

## Installation
The package is available in PyPI registry, and could be installed via `pip`:
```python
pip install mjinx
```

To run an examples or tests, please install the development version by running:
```python
pip install mjinx[dev]
```

## Usage
Here is the example of `mjinx` usage:

```python
from mujoco import mjx mjx
from mjinx.problem import Problem

# Initialize the robot model using MuJoCo
MJCF_PATH = "path_to_mjcf.xml"
mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
mjx_model = mjx.put_model(mj_model)

# Create instance of the problem
problem = Problem(mjx_model)

# Add tasks to track desired behavior
frame_task = FrameTask("ee_task", cost=1, gain=20, body_name="link7")
problem.add_component(frame_task)

# Add barriers to keep robot in a safety set
joints_barrier = JointBarrier("jnt_range", gain=10)
problem.add_component(joitns_barrier)

# Initialize the solver
solver = LocalIKSolver(mjx_model)

# Initializing initial condition
q0 = np.zeros(7)

# Initialize solver data
solver_data = solver.init()

# jit-compiling solve and integrate 
solve_jit = jax.jit(solver.solve)
integrate_jit = jax.jit(integrate, static_argnames=["dt"])

# === Control loop ===
for t in np.arange(0, 5, 1e-2):
    # Changing problem and compiling it
    frame_task.target_frame = np.array([0.1 * np.sin(t), 0.1 * np.cos(t), 0.1, 1, 0, 0,])
    problem_data = problem.compile()

    # Solving the instance of the problem
    opt_solution, solver_data = solve_jit(q, solver_data, problem_data)

    # Integrating
    q = integrate_jit(
        mjx_model,
        q,
        opt_solution.v_opt,
        dt,
    )
```

## Examples
The list of examples includes:
   1. `Kuka iiwa` local inverse kinematics ([single item](examples/local_ik.py), [vmap over desired trajectory](examples/local_ik_vmapped_output.py))
   2. `Kuka iiwa` global inverse kinematics ([single item](examples/global_ik.py), [vmap over desired trajectory](examples/global_ik_vmapped_output.py))
   3. `Go2` [batched squats](examples/go2_squat.py) example
   

## Contributing
We are always open for the suggestions and contributions. For contribution guidelines, see the [CONTRIBUTING.md](CONTRIBUTING.md) file. 

### TODO
The repostiory is under active development, the current plans before release are:
- [ ] Add examples for:
  - [ ] Quadrotor
  - [ ] Bipedal robot
  - [ ] An MPPI example
  - [ ] (?) Collaboration of several robots
- [ ] Add github pages
  - [ ] Extend mathematical descriptions in docstrings
- [ ] Add potential fields example

## Acknowledgement
The repository was highly inspired by [`pink`](https://github.com/stephane-caron/pink) and [`mink`](https://github.com/kevinzakka/mink). 
