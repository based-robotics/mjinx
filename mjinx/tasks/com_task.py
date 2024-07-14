#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Stéphane Caron, Simeon Nedelchev, Ivan Domrachev

"""Center of mass task implementation."""

from dataclasses import field

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx
from typing_extensions import override

from .base import Task


@jdc.pytree_dataclass
class ComTask(Task):
    dim = 3

    target_com: jnp.ndarray

    @override
    def compute_error(self, data: mjx.Data) -> jnp.ndarray:
        r"""..."""
        error = data.subtree_com[self.model.body_rootid[0]] - self.target_com
        return error
