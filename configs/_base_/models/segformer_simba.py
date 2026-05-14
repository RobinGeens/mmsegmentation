# Base SegFormer-style model definition for Simba backbone.
# Channel counts ([96, 192, 384, 512]) come from simba_l.
# SegformerHead matches the design used by the SegFormer paper for
# hierarchical transformer backbones — a per-stage 1x1 MLP, bilinear
# upsample to stride 4, concat, 1x1 fuse, classifier. ~0.6M params.
norm_cfg = dict(type="BN", requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type="Simba",
        variant="simba_cityscapes",
        simba_repo="../simba",
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.3,
        autocast_dtype="bf16",
        frozen_stages=-1,
    ),
    decode_head=dict(
        type="SegformerHead",
        in_channels=[96, 192, 384, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
