#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Frame task implementation."""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
from jaxlie import SE3
from typing_extensions import override

from ..configuration import get_frame_jacobian_local, get_transform_frame_to_world
from .base import Task


@jdc.pytree_dataclass
class FrameTask(Task):
    r""""""

    dim = SE3.tangent_dim

    frame_id: jdc.Static[int]
    target_frame: SE3

    @override
    def compute_error(self, data: mjx.Data) -> jnp.ndarray:
        r""""""
        return jnp.array(
            (
                get_transform_frame_to_world(
                    self.model,
                    data,
                    self.frame_id,
                ).inverse()
                @ self.target_frame
            ).log()
        )

    @override
    def compute_jacobian(self, data: mjx.Data) -> jnp.ndarray:
        T_bt = self.target_frame.inverse() @ get_transform_frame_to_world(
            self.model,
            data,
            self.frame_id,
        )

        def transform_log(tau):
            return (T_bt.multiply(SE3.exp(tau))).log()

        frame_jac = get_frame_jacobian_local(self.model, data, self.frame_id)
        jlog = jax.jacobian(transform_log)(jnp.zeros(self.dim))

        return -jlog @ frame_jac.T
