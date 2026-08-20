#!/bin/bash
# Trains the paper's reported configuration (model=large_model train=train_large) capped at
# 100000 steps, with auto-resume from the last checkpoint (train.py itself checks for
# {checkpoint.dirpath}/last.ckpt and resumes if present, so re-running this script after an
# interruption is always safe).
#
# Runs identically with or without SLURM:
#   - No SLURM: just `bash scripts/train_large_100k.sh` on any machine with GPUs. Runs
#     train.py directly and exits when training finishes.
#   - With SLURM: `sbatch scripts/train_large_100k.sh`. The #SBATCH lines below are a generic
#     starting point -- add --partition/--qos if your cluster requires them, and adjust
#     --time/--gres to taste. If your cluster gives jobs a wall-clock limit shorter than a full
#     100k-step run needs, this script self-chains: it queues its own successor (via
#     --dependency=afterany, so the chain continues however this link ends) before training
#     starts, so the run survives being killed by the time limit mid-training.
#
# All paths are configurable via env vars (see the defaults below) rather than hardcoded --
# adjust DATASET_ROOT/OUTPUT_ROOT for your own machine, or export them before running.
#
#SBATCH --job-name=stark_large_100k
#SBATCH --output=logs/stark_large_100k_%j.out
#SBATCH --error=logs/stark_large_100k_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:2

# Uses 2 GPUs, not train_large.yaml's default 4: on the authors' own cluster, every 4-GPU DDP
# attempt hung at the first call into Aligner.forward's _binarize_attention (a CUDA
# device-transfer stall, not a Python-level slowdown) across multiple nodes/GPU models, while a
# 2-GPU DDP run completed cleanly with no such hang. Root cause undetermined -- if you don't hit
# this, feel free to try more devices (set DEVICES below) and adjust ACCUM_STEPS accordingly to
# keep the same effective batch size (128, matching the paper's 4-GPU config).

set -e

: "${DATASET_ROOT:=./data/LibriTTS_R}"
: "${OUTPUT_ROOT:=./outputs/stark_large_100k}"
: "${DEVICES:=2}"
: "${ACCUM_STEPS:=4}"
: "${MAX_STEPS:=100000}"
: "${MAX_CHAIN_LENGTH:=10}"

export PATH="$HOME/.local/bin:$PATH"
# $SLURM_SUBMIT_DIR is only set when actually running under sbatch -- fall back to this script's
# own repo root so the exact same script works identically with or without SLURM.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

CKPT_DIR="$OUTPUT_ROOT/ckpt"
LOG_DIR="$OUTPUT_ROOT/log"
CHAIN_MARKER="$OUTPUT_ROOT/.chain_count"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# Self-chaining is only meaningful under SLURM (where a job has a wall-clock time limit) --
# skipped entirely otherwise, where this script just runs train.py directly until it finishes
# or you stop it.
successor_id=""
if [ -n "$SLURM_JOB_ID" ]; then
    chain_count=$(cat "$CHAIN_MARKER" 2>/dev/null || echo 0)
    # SLURM_RESTART_COUNT guard: if this job uses --requeue and gets preempted, SLURM restarts
    # this exact job/script from the top rather than ending it (incrementing
    # SLURM_RESTART_COUNT each time) -- only queue a successor on the original dispatch
    # (restart count 0), or every restart would queue a duplicate.
    if [ "${SLURM_RESTART_COUNT:-0}" -eq 0 ] && [ "$chain_count" -lt "$MAX_CHAIN_LENGTH" ]; then
        echo $((chain_count + 1)) > "$CHAIN_MARKER"
        successor_id=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID \
            "$REPO_ROOT/scripts/train_large_100k.sh")
        echo "=== queued successor job $successor_id (chain link $((chain_count + 1))/$MAX_CHAIN_LENGTH) ==="
    fi
fi

echo "=== training on $(hostname) ==="
uv run train.py train=train_large model=large_model train.trainer.max_steps=$MAX_STEPS \
    train.trainer.devices=$DEVICES \
    train.trainer.accumulate_grad_batches=$ACCUM_STEPS \
    preprocess.dataset.dataset_root=$DATASET_ROOT \
    train.checkpoint.dirpath=$CKPT_DIR \
    train.logger.save_dir=$LOG_DIR
exit_code=$?

if [ $exit_code -eq 0 ]; then
    # trainer.fit() only returns 0 like this because max_steps was actually reached -- nothing
    # else in this config stops training early.
    echo "=== training finished successfully (max_steps reached) ==="
    rm -f "$CHAIN_MARKER"
    if [ -n "$successor_id" ]; then
        echo "=== cancelling now-unneeded successor job $successor_id ==="
        scancel "$successor_id"
    fi
else
    echo "=== training exited with code $exit_code -- re-run this script (or, under SLURM, let the queued successor take over) to resume from the last checkpoint ==="
fi
exit $exit_code
