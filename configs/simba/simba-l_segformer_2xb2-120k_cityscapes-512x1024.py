_base_ = "./simba-l_segformer_2xb2-40k_cityscapes-512x1024.py"

# Stage-2 continuation: launch with
#   load_from=work_dirs/simba-l_segformer_2xb2-40k_cityscapes-512x1024/iter_40000.pth
# Skip the linear warmup (weights are already partially trained) and halve the
# peak LR vs. stage-1, since the model is no longer fresh. Poly decay runs
# over the full 120k window.
optim_wrapper = dict(optimizer=dict(lr=4e-5))

# The effective batch size determines the learning schedule and must be fixed for the whole run.
# Ensure that num_GPU * PER_GPU_BS remains constant.
_PER_GPU_BS = 2
_EFFECTIVE_BS = 4  # must match stage-1; changing this requires re-tuning LR/schedule
train_dataloader = dict(batch_size=_PER_GPU_BS, num_workers=8, persistent_workers=True, pin_memory=True)

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

if _sys.argv and "train.py" in _sys.argv[0]:
    _world_size = int(_os.environ.get("WORLD_SIZE", "1"))
    assert _world_size * _PER_GPU_BS == _EFFECTIVE_BS, (
        f"Stage-2 expects effective batch = {_EFFECTIVE_BS} (matches stage-1). "
        f"Got WORLD_SIZE={_world_size} x per_gpu_bs={_PER_GPU_BS} "
        f"= {_world_size * _PER_GPU_BS}. Adjust _PER_GPU_BS to compensate, "
        "or re-tune LR/schedule and update _EFFECTIVE_BS."
    )

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
