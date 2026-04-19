from __future__ import annotations

from typing import Any

import mujoco as mj
import mujoco.mjx as mjx


def allocate_mjx_data(
    mjx_model: mjx.Model,
    mj_model: Any | None,
    impl: str,
    nconmax: int,
    njmax: int,
) -> mjx.Data:
    """Allocate ``mjx.Data`` once; reused by ``configuration.update(JaxProblemData, q)``."""
    if impl == "warp":
        if mj_model is None:
            raise ValueError(
                "Warp backend requires a mujoco.MjModel. Build the problem with Problem.from_mj_model(...)."
            )
        return mjx.make_data(mj_model, impl="warp", nconmax=nconmax, njmax=njmax)
    if mj_model is not None:
        return mjx.make_data(mj_model)
    return mjx.make_data(mjx_model)


def create_mjx_model_and_data(
    mj_model: mj.MjModel,
    impl: str = "jax",
    nconmax: int = 512,
    njmax: int = 1024,
    graph_mode=None,
) -> tuple[mjx.Model, mjx.Data]:
    """
    Create MJX model and data for either the default JAX implementation or the Warp backend.

    - impl="jax": standard MJX-JAX path (no autodiff restrictions).
    - impl="warp": MJX-Warp path; requires static buffers nconmax and njmax.
      Optionally accepts graph_mode (see mujoco.mjx.warp.GraphMode).
    """
    if impl == "warp":
        # Warp requires make_data to receive the original mj.MjModel
        mx = mjx.put_model(mj_model, impl="warp", graph_mode=graph_mode)
        dx = mjx.make_data(mj_model, impl="warp", nconmax=nconmax, njmax=njmax)
    else:
        # JAX path: safe to pass the original mj.MjModel
        mx = mjx.put_model(mj_model)
        dx = mjx.make_data(mj_model)
    return mx, dx

