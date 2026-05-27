_base_ = "./simba-l_segformer512_4xb2-40k_cityscapes-512x1024.py"

# Halve peak LR vs. stage-1 (1.7e-4 -> 8.5e-5).
optim_wrapper = dict(optimizer=dict(lr=8.5e-5))

param_scheduler = [
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=120_000,
        by_epoch=False,
    ),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=120_000, val_interval=4000)
default_hooks = dict(checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=4000))
