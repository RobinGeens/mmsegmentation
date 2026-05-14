#!/usr/bin/env bash
# Train / evaluate SegFormer + Simba-L on Cityscapes
#
# Usage:
#   ./run_simba.sh prepare              # one-time: convert Simba checkpoint
#   ./run_simba.sh smoke                # forward pass on synthetic data (no dataset needed)
#   ./run_simba.sh train                # finetune
#   ./run_simba.sh resume [CHECKPOINT]  # resume training (latest in work_dir, or given path)
#   ./run_simba.sh eval   [CHECKPOINT]  # evaluate

set -euo pipefail

# --- Config ---

GPU=0
SIMBA_CONFIG="simba-l_segformer_2xb2-40k_cityscapes-512x1024.py"
SIMBA_CKPT_NAME="exp_approx/checkpoint-317.pth.tar"
PRETRAIN_NAME="exp_approx_317_backbone.pth"
RUN_NAME="simba-l_segformer_40k_cityscapes-512x1024"

# --- Config end ---

# Always run from the repository root (where this script lives).
cd "$(dirname "$(readlink -f "$0")")"

SIMBA_REPO="${SIMBA_REPO:-../simba}"
PYTHON="${PYTHON:-$SIMBA_REPO/env/bin/python}"
CONFIG="${CONFIG:-configs/simba/$SIMBA_CONFIG}"
SIMBA_CKPT="${SIMBA_CKPT:-"$SIMBA_REPO/checkpoints/$SIMBA_CKPT_NAME"}"
PRETRAIN="${PRETRAIN:-pretrain/$PRETRAIN_NAME}"
# Single source of truth for backbone init (merged into config by train/test).
SIMBA_PRETRAIN_CFG_OPTS=(--cfg-options "model.backbone.init_cfg.checkpoint=$PRETRAIN")
export SIMBA_PRETRAIN="$PRETRAIN"

export CUDA_VISIBLE_DEVICES="$GPU"
echo "[GPU] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# wandb knobs
export WANDB_PROJECT="${WANDB_PROJECT:-simba-cityscapes}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_NAME="$RUN_NAME"
echo "[wandb] project=$WANDB_PROJECT mode=$WANDB_MODE run=$WANDB_RUN_NAME"

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
        if [ ! -f "$PRETRAIN" ]; then
            echo "Pretrain not found at $PRETRAIN — running 'prepare' first."
            "$0" prepare
        fi
        "$PYTHON" tools/smoke_test_simba.py "$CONFIG" "$PRETRAIN"
        ;;

    train)
        echo "[train] config=$CONFIG"
        if [ ! -f "$PRETRAIN" ]; then
            echo "Pretrain not found at $PRETRAIN — running 'prepare' first."
            "$0" prepare
        fi
        "$PYTHON" tools/train.py "$CONFIG" "${SIMBA_PRETRAIN_CFG_OPTS[@]}" "$@"
        ;;

    resume)
        # Resume from the latest checkpoint in work_dirs/<config>/.
        # Optional: pass an explicit checkpoint path as the first arg.
        ckpt="${1:-}"
        work_dir="work_dirs/$(basename "$CONFIG" .py)"
        if [ -z "$ckpt" ]; then
            echo "[resume] auto-resuming from latest checkpoint in $work_dir"
            "$PYTHON" tools/train.py "$CONFIG" "${SIMBA_PRETRAIN_CFG_OPTS[@]}" --resume
        else
            echo "[resume] resuming from $ckpt"
            "$PYTHON" tools/train.py "$CONFIG" "${SIMBA_PRETRAIN_CFG_OPTS[@]}" \
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
        "$PYTHON" tools/test.py "$CONFIG" "$ckpt" "${SIMBA_PRETRAIN_CFG_OPTS[@]}"
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
