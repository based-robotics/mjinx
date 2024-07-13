#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Frame task implementation."""

from dataclasses import field

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
from typing_extensions import override

from .base import Task


@jdc.pytree_dataclass
class PositionTask(Task):
    r"""Regulate the pose of a robot frame in the world frame.

    Attributes:
        frame: Frame name, typically the name of a link or joint from the robot
            description.
        transform_target_to_world: Target pose of the frame.

    Costs are designed so that errors with varying SI units, here position and
    orientation displacements, can be cast to homogeneous values. For example,
    if task "foo" has a position cost of 1.0 and task "bar" a position cost of
    0.1, then a 1 [cm] error in task "foo" costs as much as a 10 [cm] error in
    task "bar".

    Note:
        Dimensionally, the 6D cost vector is a (normalized) force screw and our
        objective function is a (normalized) energy.
    """

    frame_id: int
    target_pos: jnp.ndarray

    @override
    def compute_error(self, model: mjx.Model, data: mjx.Data) -> jnp.ndarray:
        r""""""
        return data.xpos[self.frame_id] - self.target_pos

    def __repr__(self):
        """Human-readable representation of the task."""
        return (
            "FrameTask("
            f"frame_id={self.frame_id}, "
            f"gain={self.gain}, "
            f"cost={self.cost}, "
            f"target_pos={self.target_pos}, "
        )
