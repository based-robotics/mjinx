#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Frame task implementation."""

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
from jaxlie import SE3
from typing_extensions import override

from ..configuration import get_transform_frame_to_world
from .base import Task


@jdc.pytree_dataclass
class FrameTask(Task):
    r""""""

    frame_id: int
    target_frame: SE3

    @override
    def compute_error(self, model: mjx.Model, data: mjx.Data) -> jnp.ndarray:
        r""""""
        return jnp.array(
            (
                self.target_frame.inverse()
                @ get_transform_frame_to_world(
                    model,
                    data,
                    self.frame_id,
                )
            ).log()
        )

    @override
    def compute_jacobian(self, model: mjx.Model, data: mjx.Data) -> jnp.ndarray:
        raise NotImplementedError("Jlog6 is to be implemented.")

    def __repr__(self):
        """Human-readable representation of the task."""
        return (
            "FrameTask("
            f"frame_id={self.frame_id}, "
            f"gain={self.gain}, "
            f"orientation_cost={self.cost[:3]}, "
            f"position_cost={self.cost[3:]}, "
            f"target_frame={self.target_frame}, "
        )
