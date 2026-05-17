#!/usr/bin/env bash
# Train / evaluate SegFormer + Simba-L on Cityscapes
#
# Usage:
#   ./run_simba.sh prepare              # one-time: convert Simba checkpoint
#   ./run_simba.sh smoke                # forward pass on synthetic data (no dataset needed)
#   ./run_simba.sh train                # finetune
#   ./run_simba.sh resume [CHECKPOINT]  # resume training (latest in work_dir, or given path)
#   ./run_simba.sh eval   [CHECKPOINT]  # evaluate
#
# Two init modes for `train`, toggled by LOAD_FROM:
#   Stage-1 (LOAD_FROM=""): start from the converted Simba backbone pretrain.
#       Decoder is random. Run `prepare` once to produce the mmseg-shaped
#       backbone .pth from the upstream Simba checkpoint.
#   Stage-2 (LOAD_FROM=<ckpt>): start from a full mmseg checkpoint (backbone
#       + decoder) of a prior run. Optimizer + scheduler are NOT restored
#       (use the `resume` action for that). SIMBA_CKPT_NAME / PRETRAIN_NAME
#       are unused in this mode.

set -euo pipefail

# --- Config ---

GPU=0

TRAIN_CONFIG="simba-l_segformer_2xb2-120k_cityscapes-512x1024.py"
RUN_NAME="simba-l_segformer_120k_cityscapes-512x1024"

# Stage selector. Leave empty for stage-1 (backbone pretrain); set to a
# checkpoint path for stage-2 continuation from a prior run.
LOAD_FROM="work_dirs/simba-l_segformer_2xb2-40k_cityscapes-512x1024/iter_40000.pth"

# Stage-1 only: upstream Simba checkpoint -> mmseg-shaped backbone .pth.
SIMBA_CKPT_NAME="exp_approx/checkpoint-317.pth.tar"
PRETRAIN_NAME="exp_approx_317_backbone.pth"

# --- Config end ---

# Always run from the repository root (where this script lives).
cd "$(dirname "$(readlink -f "$0")")"

SIMBA_REPO="${SIMBA_REPO:-../simba}"
PYTHON="${PYTHON:-$SIMBA_REPO/env/bin/python}"
CONFIG="${CONFIG:-configs/simba/$TRAIN_CONFIG}"
SIMBA_CKPT="${SIMBA_CKPT:-"$SIMBA_REPO/checkpoints/$SIMBA_CKPT_NAME"}"
PRETRAIN="${PRETRAIN:-pretrain/$PRETRAIN_NAME}"
export SIMBA_PRETRAIN="$PRETRAIN"

# Build the --cfg-options array for train/eval:
#   Stage-1: inject the backbone-only pretrain via model.backbone.init_cfg.
#   Stage-2: set load_from to the full prior-run checkpoint. The backbone
#            init_cfg is irrelevant — load_from overwrites those weights.
if [ -n "${LOAD_FROM:-}" ]; then
    SIMBA_CFG_OPTS=(--cfg-options "load_from=$LOAD_FROM")
    STAGE_DESC="stage-2 continuation from $LOAD_FROM"
else
    SIMBA_CFG_OPTS=(--cfg-options "model.backbone.init_cfg.checkpoint=$PRETRAIN")
    STAGE_DESC="stage-1 from backbone pretrain $PRETRAIN"
fi
echo "[init] $STAGE_DESC"

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[GPU] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# wandb knobs
export WANDB_PROJECT="${WANDB_PROJECT:-simba-cityscapes}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_NAME="$RUN_NAME"
echo "[wandb] project=$WANDB_PROJECT mode=$WANDB_MODE run=$WANDB_RUN_NAME"

# In stage-1 the backbone pretrain must exist on disk; auto-run `prepare`
# if it doesn't. Stage-2 ignores PRETRAIN entirely.
ensure_pretrain() {
    if [ -n "${LOAD_FROM:-}" ]; then
        return 0
    fi
    if [ ! -f "$PRETRAIN" ]; then
        echo "Pretrain not found at $PRETRAIN — running 'prepare' first."
        "$0" prepare
    fi
}

cmd="${1:-}"
shift || true

case "$cmd" in
    prepare)
        echo "[prepare] Converting $SIMBA_CKPT -> $PRETRAIN"
        mkdir -p "$(dirname "$PRETRAIN")"
        "$PYTHON" tools/model_converters/simba2mmseg.py "$SIMBA_CKPT" "$PRETRAIN"
        ;;

    smoke)
        echo "[smoke] forward pass on synthetic 512x1024 input (no dataset)"
        ensure_pretrain
        "$PYTHON" tools/smoke_test_simba.py "$CONFIG" "$PRETRAIN"
        ;;

    train)
        echo "[train] config=$CONFIG"
        ensure_pretrain
        "$PYTHON" tools/train.py "$CONFIG" "${SIMBA_CFG_OPTS[@]}" "$@"
        ;;

    resume)
        # Resume from the latest checkpoint in work_dirs/<config>/.
        # Optional: pass an explicit checkpoint path as the first arg.
        # Note: --resume restores model + optimizer + scheduler + iter
        # counter from the checkpoint, so the stage-1/stage-2 init above is
        # effectively a no-op here (the resume ckpt wins).
        ckpt="${1:-}"
        work_dir="work_dirs/$(basename "$CONFIG" .py)"
        if [ -z "$ckpt" ]; then
            echo "[resume] auto-resuming from latest checkpoint in $work_dir"
            "$PYTHON" tools/train.py "$CONFIG" "${SIMBA_CFG_OPTS[@]}" --resume
        else
            echo "[resume] resuming from $ckpt"
            "$PYTHON" tools/train.py "$CONFIG" "${SIMBA_CFG_OPTS[@]}" \
                --cfg-options "load_from=$ckpt" "resume=True"
        fi
        ;;

    eval)
        ckpt="${1:-}"
        if [ -z "$ckpt" ]; then
            # Default 1: latest iter checkpoint inside the work_dir.
            work_dir="work_dirs/$(basename "$CONFIG" .py)"
            ckpt="$(ls -1 "$work_dir"/iter_*.pth 2>/dev/null | sort -V | tail -1 || true)"
            if [ -n "$ckpt" ]; then
                echo "[eval] no checkpoint passed -> latest in work_dir: $ckpt"
            elif [ -f "$PRETRAIN" ]; then
                # Default 2: the converted Simba backbone (random decoder, sanity check).
                ckpt="$PRETRAIN"
                echo "[eval] no checkpoint passed and no work_dir ckpt -> $ckpt"
            else
                echo "No checkpoint provided, none in $work_dir, and $PRETRAIN missing."
                echo "Run '$0 prepare' or pass an explicit checkpoint path."
                exit 1
            fi
        fi
        echo "[eval] config=$CONFIG  ckpt=$ckpt"
        "$PYTHON" tools/test.py "$CONFIG" "$ckpt" "${SIMBA_CFG_OPTS[@]}"
        ;;

    *)
        echo "Usage: $0 {prepare|smoke|train|resume|eval} [args...]"
        echo
        echo "  prepare               convert ../simba checkpoint -> $PRETRAIN"
        echo "  smoke                 forward pass on synthetic input (no dataset)"
        echo "  train                 finetune SegFormer+Simba on Cityscapes"
        echo "  resume [CHECKPOINT]   resume training (default: latest in work_dirs/)"
        echo "  eval   [CHECKPOINT]   evaluate (default: latest in work_dirs/)"
        exit 1
        ;;
esac
