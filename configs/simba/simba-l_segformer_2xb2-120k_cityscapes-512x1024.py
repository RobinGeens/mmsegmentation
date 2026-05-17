_base_ = "./simba-l_segformer_2xb2-40k_cityscapes-512x1024.py"

# Stage-2 continuation: launch with
#   load_from=work_dirs/simba-l_segformer_2xb2-40k_cityscapes-512x1024/iter_40000.pth
# Skip the linear warmup (weights are already partially trained) and halve the
# peak LR vs. stage-1, since the model is no longer fresh. Poly decay runs
# over the full 120k window.
optim_wrapper = dict(optimizer=dict(lr=4e-5))

param_scheduler = [
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=120000,
        by_epoch=False,
    ),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=120000, val_interval=4000)
default_hooks = dict(checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=4000))

import os as _os  # noqa: E402

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(
            project=_os.environ.get("WANDB_PROJECT", "simba-cityscapes"),
            name=_os.environ.get(
                "WANDB_RUN_NAME",
                "simba-l_segformer_120k-continue_cityscapes-512x1024",
            ),
            tags=["simba", "segformer", "cityscapes", "continue"],
        ),
    ),
]
visualizer = dict(type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer")
