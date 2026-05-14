_base_ = './simba-l_segformer_2xb2-40k_cityscapes-512x1024.py'

# Stretch the warmup + poly schedule from 40k -> 80k iterations.
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    ),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000))
