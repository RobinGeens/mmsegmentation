# Copyright (c) OpenMMLab. All rights reserved.
"""Convert a Simba (timm-style) checkpoint to MMSegmentation backbone format.

The upstream Simba checkpoints (e.g.
``simba/checkpoints/simba_l_bf16_TL/last.pth.tar``) contain:

  - ``patch_embed{i}.*``, ``block{i}.*``, ``norm{i}.*``  (kept, the backbone)
  - ``post_network.*``, ``head.*``, ``aux_head.*``        (dropped, classifier)

This script strips the classifier-only keys and saves a flat state-dict
ready to be loaded via ``init_cfg=dict(type='Pretrained', checkpoint=...)``
in the mmseg Simba backbone.

Usage:
    python tools/model_converters/simba2mmseg.py \\
        path_to_simba_checkpoint \\
        pretrain/checkpoint_name.pth
"""
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch

KEEP_PREFIXES = ('patch_embed', 'block', 'norm')


def convert_simba(state_dict):
    new_sd = OrderedDict()
    dropped = []
    for k, v in state_dict.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('backbone.'):
            nk = nk[len('backbone.'):]
        if nk.startswith(KEEP_PREFIXES):
            new_sd[nk] = v
        else:
            dropped.append(nk)
    return new_sd, dropped


def main():
    parser = argparse.ArgumentParser(
        description='Convert Simba pretrained checkpoint to MMSegmentation '
        'backbone format.')
    parser.add_argument('src', help='source checkpoint path')
    parser.add_argument('dst', help='destination path')
    args = parser.parse_args()

    # weights_only=False because upstream simba checkpoints contain an
    # argparse.Namespace under the 'args' key.
    ckpt = torch.load(args.src, map_location='cpu', weights_only=False)
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif 'model' in ckpt:
        sd = ckpt['model']
    else:
        sd = ckpt

    new_sd, dropped = convert_simba(sd)
    print(f'Kept {len(new_sd)} backbone keys, dropped {len(dropped)} '
          f'classifier keys: {dropped[:8]}{"..." if len(dropped) > 8 else ""}')

    mmengine.mkdir_or_exist(osp.dirname(args.dst) or '.')
    torch.save({'state_dict': new_sd}, args.dst)
    print(f'Saved to {args.dst}')


if __name__ == '__main__':
    main()
