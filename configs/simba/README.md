# Simba (Mamba + EinFFT) for semantic segmentation

This directory contains configs that pair the **SiMBA** backbone with a
mmsegmentation decoder. The backbone wrapper at
[mmseg/models/backbones/simba.py](../../mmseg/models/backbones/simba.py)
imports the actual model definition from `../simba/simba/simba_bf16.py`
(the sibling repo) and exposes it as a 4-stage feature extractor with
strides `(4, 8, 16, 32)` and channels `[96, 192, 384, 512]`.

The decoder is **SegformerHead** — designed for hierarchical transformer
backbones, ~0.6 M params (vs ~30 M for UPerHead), so the full model is
~34 M (33.5 M backbone + 0.6 M head + tiny preprocessor).

## Files

| Config                                                                                                   | Purpose                                                     |
| -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| [`simba-l_segformer_2xb2-40k_cityscapes-512x1024.py`](simba-l_segformer_2xb2-40k_cityscapes-512x1024.py) | SegFormer + Simba-L on Cityscapes, 40k iters, 512x1024 crop |
| [`simba-l_segformer_2xb2-80k_cityscapes-512x1024.py`](simba-l_segformer_2xb2-80k_cityscapes-512x1024.py) | Same, 80k iters                                             |
| [`../_base_/models/segformer_simba.py`](../_base_/models/segformer_simba.py)                             | Reusable SegFormer+Simba model definition                   |
| [`../../tools/model_converters/simba2mmseg.py`](../../tools/model_converters/simba2mmseg.py)             | Strip classifier-only keys from a Simba checkpoint          |

## Environment

Directly inherited from `../simba/env`.

## 1. Prepare the pretrained backbone

Convert the upstream classification checkpoint to a backbone-only
state-dict that the mmseg `Simba` backbone can load:

```bash
./run_simba.sh prepare
```

Drops `head`, `aux_head`, and `post_network` (552 backbone keys kept).
The default output path is `pretrain`

## 2. Cityscapes data

Place the Cityscapes dataset at `data/cityscapes/` with the standard
`leftImg8bit/` and `gtFine/` layout, then generate the train IDs:

```bash
python -m cityscapesscripts.preparation.createTrainIdLabelImgs
```

If you don't have the data yet and have an account, the easiest route
is to add a `machine cityscapes-dataset.com login <user> password <pw>`
entry to `~/.netrc` and have the script bridge it to `csDownload` for
you (see `run_simba.sh`).

## 3. Evaluate (zero-shot from the converted backbone)

This runs the model with the pretrained Simba backbone but a randomly
initialised SegformerHead — useful for sanity-checking the backbone
forward and the data pipeline. Real numbers come after finetuning.

```bash
python tools/test.py \
    configs/simba/simba-l_segformer_2xb2-40k_cityscapes-512x1024.py \
    pretrain/simba_l_bf16_TL_backbone.pth
```

To evaluate a finetuned mmseg checkpoint instead, point the second
argument at the finetuned `.pth` (or just call `./run_simba.sh eval` and
let it pick up the latest in `work_dirs/`).

## 4. Finetune backbone + decoder on Cityscapes

```bash
python tools/train.py \
    configs/simba/simba-l_segformer_2xb2-40k_cityscapes-512x1024.py
```

## Caveats

- The CUDA kernels (`mamba_ssm`, `causal_conv1d`) are required to run
  the model — there is no CPU path. Make sure the env from `../simba`
  works before launching mmseg training.
- The classifier-only modules (`head`, `aux_head`, `post_network`) are
  intentionally dropped during conversion; they are not used for
  segmentation.
- The default config assumes `data/cityscapes/` and
  `pretrain/simba_l_bf16_TL_backbone.pth` exist. Override them via
  `--cfg-options` if you keep them elsewhere.
