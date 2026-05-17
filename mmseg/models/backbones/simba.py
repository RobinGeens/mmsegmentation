# Copyright (c) OpenMMLab. All rights reserved.
"""Simba backbone for mmsegmentation.

This file wraps the SiMBA model defined in the sibling repository at
``../simba`` and exposes it as an mmseg backbone that produces 4 multi-scale
feature maps (after each of the 4 stages), suitable for decoders such as
UPerHead.

Notes
-----
* The actual model definition lives in ``<simba_repo>/simba/simba_bf16.py``.
  We import it lazily and add the directory to ``sys.path`` because the
  upstream files use bare module imports (e.g. ``from quantizer_basic
  import ...``).
* CUDA kernels (mamba_ssm + causal_conv1d) are required for execution.
"""
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader

from mmseg.registry import MODELS


def _ensure_simba_on_path(simba_repo: str) -> None:
    """Prepend ``simba_repo`` (git root) to ``sys.path``.

    Upstream code imports the inner package as ``simba.*`` (e.g.
    ``from simba.quantizer_basic import ...``), so the path entry must be
    the repo root, not ``<simba_repo>/simba``.
    """
    pkg_dir = os.path.join(simba_repo, "simba")
    if not os.path.isdir(pkg_dir):
        raise FileNotFoundError(
            f"Simba package directory not found: {pkg_dir}. " f"Pass `simba_repo=<path>` to the backbone or symlink it."
        )
    if simba_repo not in sys.path:
        sys.path.insert(0, simba_repo)


@MODELS.register_module()
class Simba(BaseModule):
    """SiMBA backbone (Mamba + EinFFT) for dense prediction.

    Produces 4 feature maps with strides ``(4, 8, 16, 32)`` and channel
    counts matching the ``embed_dims`` of the chosen variant.

    Args:
        variant (str): Name of the factory function in ``simba_bf16``.
            Default: ``'simba_cityscapes'``.
        simba_repo (str): Path to the simba repo root (the directory that
            contains the ``simba`` package). The default assumes the layout
            ``<workspace>/mmsegmentation`` and ``<workspace>/simba``.
        out_indices (Sequence[int]): Stage indices to output.
        drop_path_rate (float): Stochastic depth rate.
        autocast_dtype (str | None): If set (``'bf16'`` / ``'fp16'``), wraps
            the forward pass in ``torch.amp.autocast`` with this dtype to
            match the regime the weights were trained in.
        frozen_stages (int): Stages to freeze (``-1`` = none, ``0`` = stem
            only, ``i`` = stem + first ``i`` stages).
        init_cfg (dict | None): Standard mmengine init config. Passing
            ``dict(type='Pretrained', checkpoint=...)`` loads a converted
            mmseg-style checkpoint via ``init_weights``.
    """

    _VARIANT_DIMS = {
        "simba_cityscapes": [96, 192, 384, 512],
    }

    def __init__(
        self,
        variant: str = "simba_cityscapes",
        simba_repo: str = "../simba",
        out_indices=(0, 1, 2, 3),
        drop_path_rate: float = 0.3,
        autocast_dtype: str = "bf16",
        frozen_stages: int = -1,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        # Resolve simba_repo relative to CWD if it is relative.
        simba_repo = os.path.abspath(simba_repo)
        _ensure_simba_on_path(simba_repo)

        from simba import simba_bf16  # noqa: PLC0415 — after sys.path fixup

        if not hasattr(simba_bf16, variant):
            raise ValueError(
                f"Unknown Simba variant {variant!r}. Available: "
                f'{[n for n in dir(simba_bf16) if n.startswith("simba_")]}'
            )

        # The upstream factory returns a SiMBA module configured with the
        # right dtype kwargs. token_label=True adds the post_network/aux_head
        # branch which is irrelevant for segmentation; we strip it below.
        full_model = getattr(simba_bf16, variant)(
            pretrained=False,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            token_label=False,
        )

        self.num_stages = full_model.num_stages
        self.embed_dims = self._VARIANT_DIMS.get(variant)
        self.out_indices = tuple(out_indices)
        self.frozen_stages = frozen_stages

        # Move the per-stage submodules into ``self`` so that their parameter
        # names mirror the upstream checkpoint (``patch_embed{i}``,
        # ``block{i}``, ``norm{i}``). This keeps the converted state-dict
        # keys readable.
        for i in range(self.num_stages):
            for prefix in ("patch_embed", "block", "norm"):
                name = f"{prefix}{i + 1}"
                self.add_module(name, getattr(full_model, name))

        # Discard classifier-only modules (they are not present after
        # ``token_label=False`` for some, but be defensive).
        for attr in ("head", "aux_head", "post_network"):
            if hasattr(full_model, attr):
                delattr(full_model, attr)

        self._autocast_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            None: None,
        }.get(autocast_dtype, None)

        self._freeze_stages()

    # ------------------------------------------------------------------
    # Freeze handling
    # ------------------------------------------------------------------
    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return
        # Freeze stem (patch_embed1) when frozen_stages >= 0.
        if self.frozen_stages >= 0:
            m = getattr(self, "patch_embed1")
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
        # Freeze first ``frozen_stages`` stages entirely.
        for i in range(1, self.frozen_stages + 1):
            for prefix in ("block", "norm", "patch_embed"):
                if prefix == "patch_embed" and i == 1:
                    continue  # already done above
                name = f"{prefix}{i}"
                if hasattr(self, name):
                    m = getattr(self, name)
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        return self

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------
    def init_weights(self):
        # Skip the backbone-only pretrain load when there's no checkpoint to
        # load. This covers two cases: (1) init_cfg is unset, and (2) init_cfg
        # is present but checkpoint is None — e.g. stage-2 continuation where
        # the top-level `load_from` provides the full model weights and any
        # backbone-only init would be immediately overwritten.
        ckpt_path = (self.init_cfg or {}).get("checkpoint")
        if not ckpt_path:
            print_log(
                f"No pre-trained weights for {self.__class__.__name__}, " f"training from scratch", logger="current"
            )
            return
        ckpt = CheckpointLoader.load_checkpoint(ckpt_path, logger="current", map_location="cpu")

        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt

        # Strip a few common prefixes.
        cleaned = OrderedDict()
        for k, v in sd.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("backbone."):
                nk = nk[len("backbone.") :]
            cleaned[nk] = v

        # Filter classifier-only keys that don't exist on the backbone.
        own_keys = set(self.state_dict().keys())
        filtered = OrderedDict((k, v) for k, v in cleaned.items() if k in own_keys)
        dropped = [k for k in cleaned.keys() if k not in own_keys]
        if dropped:
            print_log(
                f"Simba: dropping {len(dropped)} keys not in backbone " f"(e.g. {dropped[:3]}...)", logger="current"
            )

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print_log(
            f"Simba: loaded checkpoint {ckpt_path} | " f"missing={len(missing)} unexpected={len(unexpected)}",
            logger="current",
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _forward_features(self, x: torch.Tensor):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            blocks = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in blocks:
                x = blk(x, H, W)

            x_normed = norm(x)
            x_2d = x_normed.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outs.append(x_2d)
            if i != self.num_stages - 1:
                # Feed the 2D normed feature into the next downsampling stage.
                x = x_2d
        return outs

    def forward(self, x: torch.Tensor):
        if self._autocast_dtype is not None and x.is_cuda:
            with torch.amp.autocast("cuda", dtype=self._autocast_dtype):
                return self._forward_features(x)
        return self._forward_features(x)
