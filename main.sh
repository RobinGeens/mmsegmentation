#!/usr/bin/env bash
# Train / evaluate SegFormer + Simba-L on Cityscapes.
#
# Usage:
#   ./run_simba.sh prepare       one-time: convert upstream Simba ckpt -> mmseg backbone .pth
#   ./run_simba.sh smoke         forward pass on synthetic input (no dataset)
#   ./run_simba.sh train         train; init source picked by $MODE (see config block)
#   ./run_simba.sh resume        shorthand for MODE=continue
#   ./run_simba.sh eval [CKPT]   evaluate (default: latest ckpt in this config's work_dir)

set -euo pipefail

# --- Config ---

# Comma-separated GPU ids. Multi-GPU launches via torch.distributed. Per-GPU
# batch size lives in the config (train_dataloader.batch_size); changing the
# GPU count here without revisiting that will change effective batch size.
GPUS="0,1,2,3"
NUM_GPUS=$(awk -F',' '{print NF}' <<< "$GPUS")

TRAIN_CONFIG="simba-l_segformer512_4xb2-120k_cityscapes-512x1024.py"
RUN_NAME="simba-l_segformer512_4xb2-120k_cityscapes-512x1024"

# How to start training. Pick exactly one:
#
#   backbone — Init backbone from the converted Simba pretrain; segmentation
#              head is random. Optimizer + scheduler start fresh at iter 0.
#
#   seed     — Init the full model (backbone + head) from a prior mmseg checkpoint at SEED_CKPT. 
#              Optimizer + scheduler start fresh at iter 0. Writes to this config's work_dir, not SEED_CKPT's.
#              Use this to bridge between two configs (e.g. extend a 40k schedule with a fresh 120k schedule).
#
#   continue — Resume the latest checkpoint inside this config's own
#              work_dir, restoring optimizer + scheduler + iter counter.
#              Automatically discovers the latest checkpoint in work_dir.
MODE="seed"

# Used only when MODE=seed.
SEED_CKPT="work_dirs/simba-l_segformer512_4xb2-40k_cityscapes-512x1024/iter_40000.pth"
# Used only when MODE=backbone
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
WORK_DIR="work_dirs/$(basename "$CONFIG" .py)"
export SIMBA_PRETRAIN="$PRETRAIN"

export CUDA_VISIBLE_DEVICES="$GPUS"
echo "[GPU] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (n=$NUM_GPUS)"

# Build the train launch prefix. Distributed launch when NUM_GPUS>1, plain
# python otherwise. Per-mode --cfg-options are appended downstream.
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH=("$PYTHON" -m torch.distributed.launch
            --nnodes=1 --nproc_per_node="$NUM_GPUS"
            --master_port="${PORT:-29500}"
            tools/train.py "$CONFIG" --launcher pytorch)
else
    LAUNCH=("$PYTHON" tools/train.py "$CONFIG")
fi

export WANDB_PROJECT="${WANDB_PROJECT:-simba-cityscapes}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_NAME="$RUN_NAME"
echo "[wandb] project=$WANDB_PROJECT mode=$WANDB_MODE run=$WANDB_RUN_NAME"

ensure_pretrain() {
    if [ ! -f "$PRETRAIN" ]; then
        echo "Pretrain not found at $PRETRAIN — running 'prepare' first."
        "$0" prepare
    fi
}

do_train() {
    case "$MODE" in
        backbone)
            ensure_pretrain
            echo "[train:backbone] backbone <- $PRETRAIN  (head random, optim fresh, work_dir=$WORK_DIR)"
            "${LAUNCH[@]}" \
                --cfg-options "model.backbone.init_cfg.checkpoint=$PRETRAIN" "$@"
            ;;
        seed)
            if [ ! -f "$SEED_CKPT" ]; then
                echo "MODE=seed but SEED_CKPT does not exist: $SEED_CKPT" >&2
                exit 1
            fi
            echo "[train:seed] model <- $SEED_CKPT  (optim fresh, work_dir=$WORK_DIR)"
            "${LAUNCH[@]}" \
                --cfg-options "load_from=$SEED_CKPT" "$@"
            ;;
        continue)
            # No --cfg-options: with load_from unset, mmengine auto-discovers
            # work_dir/last_checkpoint. Setting load_from would override that.
            echo "[train:continue] auto-resume latest ckpt in $WORK_DIR"
            "${LAUNCH[@]}" --resume "$@"
            ;;
        *)
            echo "Unknown MODE: '$MODE' (expected: backbone | seed | continue)" >&2
            exit 1
            ;;
    esac
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
        echo "[smoke] forward pass on synthetic 512x1024 input"
        ensure_pretrain
        "$PYTHON" tools/smoke_test_simba.py "$CONFIG" "$PRETRAIN"
        ;;

    train)
        do_train "$@"
        ;;

    resume)
        MODE=continue
        do_train "$@"
        ;;

    eval)
        ckpt="${1:-}"
        if [ -z "$ckpt" ]; then
            ckpt="$(ls -1 "$WORK_DIR"/iter_*.pth 2>/dev/null | sort -V | tail -1 || true)"
            if [ -n "$ckpt" ]; then
                echo "[eval] no ckpt passed -> latest in work_dir: $ckpt"
            elif [ -f "$PRETRAIN" ]; then
                ckpt="$PRETRAIN"
                echo "[eval] no work_dir ckpt -> falling back to backbone pretrain: $ckpt"
            else
                echo "No checkpoint provided, none in $WORK_DIR, and $PRETRAIN missing." >&2
                echo "Run '$0 prepare' or pass an explicit checkpoint path." >&2
                exit 1
            fi
        fi
        echo "[eval] config=$CONFIG  ckpt=$ckpt"
        "$PYTHON" tools/test.py "$CONFIG" "$ckpt"
        ;;

    *)
        echo "Usage: $0 {prepare|smoke|train|resume|eval} [args...]"
        echo
        echo "  prepare               convert upstream Simba ckpt -> $PRETRAIN"
        echo "  smoke                 forward pass on synthetic input (no dataset)"
        echo "  train                 train; init source picked by \$MODE (currently: $MODE)"
        echo "  resume                shorthand for MODE=continue"
        echo "  eval [CHECKPOINT]     evaluate (default: latest in $WORK_DIR)"
        exit 1
        ;;
esac
