# Segmentation Head Options for Simba-L on Cityscapes

Backbone output: 4 stages with channels `[96, 192, 384, 512]`, strides `(4, 8, 16, 32)`.
Current result: **79.2 mIoU** with SegformerHead (channels=256).

## Head Comparison

Computed with `in_channels=[96, 192, 384, 512]`, `num_classes=19`.

| Head                        | channels | Params   | Inputs     | Building blocks                                                                                                                                                                                                                                                             |
| --------------------------- | -------- | -------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SegformerHead** (current) | 256      | **0.6M** | all stages | 4x 1x1 **conv**+BN per stage, bilinear upsample, concat, 1x1 **conv**+BN fuse, 1x1 **conv** classifier. Pure MLP/conv, no attention.                                                                                                                                        |
| SegformerHead               | 512      | 1.7M     | all stages | same architecture, wider                                                                                                                                                                                                                                                    |
| SegformerHead               | 768      | 3.3M     | all stages | same architecture, widest practical                                                                                                                                                                                                                                         |
| **UPerHead**                | 256      | **8.4M** | all stages | **PSP** module on stage 4 (4x adaptive-avg-pool + 1x1 **conv**+BN, 3x3 **conv**+BN bottleneck) + **FPN** (1x1 lateral **conv**+BN + 3x3 **conv**+BN per stage, top-down upsample+add), final 3x3 **conv**+BN fuse, 1x1 **conv** classifier. All convolutions, no attention. |
| UPerHead                    | 512      | 29.7M    | all stages | same architecture, wider (standard for Swin-L / ConvNeXt-L)                                                                                                                                                                                                                 |
| FPNHead                     | 256      | 4.5M     | all stages | Per-scale chain of 3x3 **conv**+BN + bilinear 2x upsample (repeated log2(stride/4) times), sum across scales, 1x1 **conv** classifier. All convolutions.                                                                                                                    |
| FPNHead                     | 512      | 12.6M    | all stages | same architecture, wider                                                                                                                                                                                                                                                    |
| PSPHead                     | 512      | 12.9M    | stage 4    | 4x adaptive-avg-pool at scales (1,2,3,6) + 1x1 **conv**+BN, concat with input, 3x3 **conv**+BN bottleneck, 1x1 **conv** classifier. All convolutions + pooling.                                                                                                             |
| DAHead                      | 512      | 9.8M     | stage 4    | Parallel spatial **self-attention** (position) + channel **self-attention** (CAM), each preceded by 3x3 **conv**+BN. Outputs summed, 3x3 **conv**+BN, 1x1 **conv** classifier. Two attention branches.                                                                      |
| OCRHead                     | 512      | 3.6M     | stage 4    | Soft object regions via 1x1 **conv** (coarse seg), per-class weighted-pool to get K prototypes, pixel-to-prototype **cross-attention** (key/query/value as 1x1 **conv**), concat with input, 1x1 **conv**+BN fuse, 1x1 **conv** classifier.                                 |
| FCNHead (auxiliary)         | 256      | 0.9M     | stage 3    | Single 1x1 **conv**+BN+ReLU, 1x1 **conv** classifier. Used as auxiliary loss (loss_weight=0.4), not as primary head.                                                                                                                                                        |

## Recommendations (ranked by expected impact)

### 1. UPerHead + FCN auxiliary — strongest upgrade (~+2-4 mIoU)

The de-facto standard for hierarchical backbones (Swin, ConvNeXt, BEiT all use it).
Combines FPN-style lateral connections with Pyramid Pooling on the deepest stage.
Adding an FCN auxiliary head on stage 3 (loss_weight=0.4) provides deep supervision
that stabilizes training.

- `UPerHead(channels=512)` + `FCNHead aux` = ~30.6M head params
- `UPerHead(channels=256)` + `FCNHead aux` = ~9.3M head params (good starting point)

Reference: Swin-T + UPerNet@512 = 81.2 mIoU on Cityscapes (similar backbone size).

### 2. Widen current SegformerHead — small/easy upgrade (~+0.5-1 mIoU)

Increase `channels` from 256 → 512 (1.7M) or 768 (3.3M). Minimal config change,
keeps the lightweight MLP character. Diminishing returns beyond 512.

### 3. FPNHead — middle ground (~+1-2 mIoU)

Progressive upsampling with 3×3 convs at each scale. Heavier than SegformerHead
but lighter than UPerHead. No PSP pooling → less global context than UPerHead.

### 4. OCRHead — efficient attention (~+1-2 mIoU)

Object-Contextual Representations: computes per-class prototypes and refines each
pixel by attending to its class center. Only 3.6M params but uses a single stage.
Combining with an FCN auxiliary head is recommended.

### 5. DAHead — dual attention (~+1-2 mIoU)

Spatial + channel attention in parallel on the deepest stage. 9.8M params, only
uses stage 4. Good context modeling but single-scale limits it vs UPerHead.

## Crop Size Options

Current: **512x1024** (train crop and slide-window test crop).

Simba uses a partitioned DFT internally. Each token-sequence length (per stage)
must have an entry in `DFT_PARTITIONS` in `simba_bf16.py`, otherwise it falls back
to L=1 (single giant DFT matrix -> OOM). The "padded" column shows the next
power-of-2 that the token count is padded to before the DFT.

| Crop                   | Stage 1 (/4)             | Stage 2 (/8)             | Stage 3 (/16)       | Stage 4 (/32)       | Total tokens |
| ---------------------- | ------------------------ | ------------------------ | ------------------- | ------------------- | ------------ |
| 512x512                | 16384 (pad 16384) **NO** | 4096 (pad 4096) yes      | 1024 (pad 1024) yes | 256 (pad 256) yes   | 21,760       |
| **512x1024** (current) | 32768 (pad 32768) yes    | 8192 (pad 8192) yes      | 2048 (pad 2048) yes | 512 (pad 512) yes   | 41,520       |
| 768x768                | 36864 (pad 65536) **NO** | 9216 (pad 16384) **NO**  | 2304 (pad 4096) yes | 576 (pad 1024) yes  | 48,960       |
| 768x1536               | 73728 (pad 131072) yes   | 18432 (pad 32768) yes    | 4608 (pad 8192) yes | 1152 (pad 2048) yes | 97,920       |
| 1024x1024              | 65536 (pad 65536) **NO** | 16384 (pad 16384) **NO** | 4096 (pad 4096) yes | 1024 (pad 1024) yes | 87,040       |

### Changing crop size: do I need to retrain?

- **Training crop:** yes, retrain or finetune. The model learns spatial patterns
  at the training resolution. Cheapest path: use the current 120k checkpoint as a
  seed (`MODE=seed` in `run_simba.sh`) and finetune for ~40k iterations at the new
  crop size.
- **Test/sliding-window crop:** can be changed without retraining, but keeping it
  matched to the training crop is standard practice. A large mismatch causes a
  distribution gap.

## Other ways to improve mIoU (beyond the head)

| Strategy               | Expected gain | Cost                       |
| ---------------------- | ------------- | -------------------------- |
| Add auxiliary FCN head | +0.5–1.0      | ~0.9M params, +40% loss    |
| Larger crop (768×1536) | +0.5–1.5      | ~2.3× memory/iter          |
| OHEM loss              | +0.3–0.8      | focuses on hard pixels     |
| Lovász-softmax loss    | +0.3–0.5      | directly optimizes IoU     |
| Multi-scale test (TTA) | +1.0–2.0      | inference-only, ~5× slower |
| Class-uniform sampling | +0.3–0.5      | improves rare-class recall |
| Longer schedule (160k) | +0.2–0.5      | more training time         |

## Quick-start: UPerHead config

To switch from SegformerHead → UPerHead, create a new base model config:

```python
# configs/_base_/models/upernet_simba.py
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True, pad_val=0, seg_pad_val=255,
)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='Simba',
        variant='simba_cityscapes',
        simba_repo='../simba',
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.3,
        autocast_dtype='bf16',
        frozen_stages=-1,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
```
