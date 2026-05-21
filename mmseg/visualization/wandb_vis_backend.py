# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import numpy as np
import torch
from mmengine.registry import VISBACKENDS
from mmengine.visualization import WandbVisBackend
from mmengine.visualization.vis_backend import force_init_env


@VISBACKENDS.register_module()
class IterStepWandbVisBackend(WandbVisBackend):
    """WandbVisBackend that plots all metrics against training ``iter``.

    Upstream ``WandbVisBackend`` discards the ``step`` argument and relies on
    wandb's auto-incrementing internal step. That makes resumed/seeded runs
    overlap on the X-axis even though their training iter ranges differ.

    This backend (a) injects ``iter`` into every logged dict using the step
    LoggerHook passes, and (b) calls ``wandb.define_metric("*", step_metric=
    "iter")`` on init so wandb plots every metric against that field.
    """

    def _init_env(self):
        super()._init_env()
        self._wandb.define_metric('iter')
        self._wandb.define_metric('*', step_metric='iter')

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        self._wandb.log({name: value, 'iter': step}, commit=self._commit)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        payload = dict(scalar_dict)
        payload.setdefault('iter', step)
        self._wandb.log(payload, commit=self._commit)
