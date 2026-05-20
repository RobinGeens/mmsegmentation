_base_ = [
    "../_base_/models/segformer_simba.py",
    "../_base_/datasets/cityscapes.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]

crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)

# Checkpoint path is overridden by `run_simba.sh`.
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=None),
        variant="simba_cityscapes",
        simba_repo="../simba",
        drop_path_rate=0.3,
        autocast_dtype="bf16",
    ),
    decode_head=dict(in_channels=[96, 192, 384, 512], num_classes=19),
    # Slide eval: whole-image inference would feed stage 1 a 512x256=131072
    # token sequence, which has no DFT_PARTITIONS entry and degenerates to an
    # L=1 partitioning (262144x262144 real DFT matrix -> OOM). Sliding at the
    # training crop keeps every per-window N matched to a partition entry.
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(384, 768)),
)

# AdamW with linear-warmup poly schedule, decoder LR > backbone LR. The
# backbone is highly mixed-precision (bf16 internals + custom quantizers),
# so we keep the backbone LR small for finetuning.
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=8.5e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1, decay_mult=1.0),
            "norm": dict(decay_mult=0.0),
            "A_log": dict(decay_mult=0.0),
            "D": dict(decay_mult=0.0),
            "complex_bias_1": dict(decay_mult=0.0),
            "complex_bias_2": dict(decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    ),
]

# Single-GPU Cityscapes train.
train_dataloader = dict(batch_size=4, num_workers=20)
val_dataloader = dict(batch_size=1, num_workers=20)
test_dataloader = val_dataloader

# wandb logging. Override the run name / project via env vars in run_simba.sh
# (WANDB_PROJECT, WANDB_RUN_NAME) or by editing this dict directly.
import os as _os  # noqa: E402

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(
            project=_os.environ.get("WANDB_PROJECT", "simba-cityscapes"),
            name=_os.environ.get("WANDB_RUN_NAME", "simba-l_segformer_40k_cityscapes-512x1024"),
            tags=["simba", "segformer", "cityscapes"],
        ),
    ),
]
visualizer = dict(type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer")
