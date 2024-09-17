#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Stéphane Caron, Simeon Nedelchev, Ivan Domrachev

"""Center of mass task implementation."""

from typing import Callable, Sequence, final

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco.mjx as mjx

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxComTask(JaxTask):
    target_com: jnp.ndarray

    @final
    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        r"""..."""
        error = data.subtree_com[self.model.body_rootid[0], self.mask_idxs].ravel() - self.target_com
        return error


class ComTask(Task[JaxComTask]):
    __target_com: jnp.ndarray

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Sequence | None = None,
    ):
        if mask is not None and len(mask) != 3:
            raise ValueError("provided mask is too large, expected 1D vector of length 3")
        super().__init__(name, cost, gain, gain_fn, lm_damping, mask=mask)
        self._dim = 3 if mask is None else len(self.mask_idxs)
        self.target_com = jnp.zeros(self._dim)

    @property
    def target_com(self) -> jnp.ndarray:
        return self.__target_com

    @target_com.setter
    def target_com(self, value: Sequence):
        self.update_target_com(value)

    def update_target_com(self, target_com: Sequence):
        target_com_jnp = jnp.array(target_com)
        if target_com_jnp.shape[-1] != self._dim:
            raise ValueError(
                "invalid last dimension of target CoM : " f"{target_com_jnp.shape[-1]} given, expected {self._dim} "
            )
        self._modified = True
        self.__target_com = target_com_jnp

    @final
    def _build_component(self) -> JaxComTask:
        return JaxComTask(
            dim=self.dim,
            model=self.model,
            cost=self.matrix_cost,
            gain=self.vector_gain,
            gain_function=self.gain_fn,
            lm_damping=self.lm_damping,
            target_com=self.target_com,
            mask_idxs=self.mask_idxs,
        )
