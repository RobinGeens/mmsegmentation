"""Smoke-test the Simba+SegFormer config end-to-end on a synthetic image.

Builds the full model from the config, loads the converted backbone
checkpoint, and runs a forward pass on random input. No Cityscapes data
required — useful when data is not yet provisioned.

Usage:
    python tools/smoke_test_simba.py CONFIG CHECKPOINT

    CHECKPOINT may be omitted if env SIMBA_PRETRAIN is set (e.g. by run_simba.sh).
"""
import os
import sys

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS

CONFIG = sys.argv[1] if len(sys.argv) > 1 else None
CKPT = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('SIMBA_PRETRAIN')

if CONFIG is None or CKPT is None:
    print(
        'Usage: python tools/smoke_test_simba.py CONFIG CHECKPOINT\n'
        '   or: SIMBA_PRETRAIN=/path/to/converted_backbone.pth '
        'python tools/smoke_test_simba.py CONFIG',
        file=sys.stderr,
    )
    sys.exit(1)

print(f'[config] {CONFIG}')
print(f'[ckpt]   {CKPT}')

cfg = Config.fromfile(CONFIG)
init_default_scope(cfg.get('default_scope', 'mmseg'))

# Drop the init_cfg so we don't try to load the backbone twice; we'll do
# it manually with `load_checkpoint` (which tolerates a flat backbone-only
# state-dict on the full model after we add the 'backbone.' prefix).
cfg.model.backbone.pop('init_cfg', None)

model = MODELS.build(cfg.model)
model.eval()

# Load backbone weights with prefix and report.
sd = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = sd.get('state_dict', sd)
prefixed = {f'backbone.{k}': v for k, v in sd.items()}
res = model.load_state_dict(prefixed, strict=False)
print(f'[load] missing={len(res.missing_keys)}  '
      f'unexpected={len(res.unexpected_keys)}')
backbone_missing = [k for k in res.missing_keys if k.startswith('backbone.')]
print(f'[load] backbone missing: {len(backbone_missing)} '
      f'(should be 0 for the converted ckpt)')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[device] {device}')
model = model.to(device)

# Synthetic Cityscapes-shaped batch.
B, C, H, W = 1, 3, 512, 1024
x = torch.randn(B, C, H, W, device=device)

# 1) Backbone-only forward.
with torch.no_grad():
    feats = model.backbone(x)
print('[backbone] output shapes:')
for i, f in enumerate(feats):
    print(f'  stage {i}: {tuple(f.shape)}  dtype={f.dtype}')

# 2) Full encoder-decoder forward (logits before upsample).
with torch.no_grad():
    seg_logits = model.decode_head(feats)
print(f'[decode_head] logits: {tuple(seg_logits.shape)} dtype={seg_logits.dtype}')

print('[ok] Simba + decoder forward pass succeeded.')
