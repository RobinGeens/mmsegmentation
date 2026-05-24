_base_ = "./simba-l_segformer_2xb2-40k_cityscapes-512x1024.py"

model = dict(
    decode_head=dict(channels=512),
)

# 4 GPUs x batch 2 = effective batch 8 (2x the original).
# Scale LR linearly: 8.5e-5 * 2 = 1.7e-4.
train_dataloader = dict(batch_size=2)
optim_wrapper = dict(optimizer=dict(lr=1.7e-4))

import os as _os  # noqa: E402

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="IterStepWandbVisBackend",
        init_kwargs=dict(
            project=_os.environ.get("WANDB_PROJECT", "simba-cityscapes"),
            name=_os.environ.get(
                "WANDB_RUN_NAME",
                "simba-l_segformer512_4xb2-40k_cityscapes-512x1024",
            ),
            tags=["simba", "segformer512", "cityscapes"],
        ),
    ),
]
visualizer = dict(type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer")
