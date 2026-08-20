#!/bin/bash
# One-off read-only quality check of a checkpoint (mid-training or final -- this does NOT push
# anything to the Hub) against the full LibriTTS-R test-clean split: PCC/DTW of predicted EMA vs
# ground truth (Setup 1 predicted-duration + Setup 2 aligner-duration, same methodology as
# eval_and_push.py / the paper's Table 2), plus DNSMOS + UTMOSv2 on the Setup 1 resynthesized
# audio vs ground truth vs oracle resynthesis (paper Table 1 methodology, extended with UTMOSv2).
#
# Usage: bash scripts/eval_full_testset.sh <ckpt_path> <results_path>
# Runs identically with or without SLURM -- direct invocation above, or
# `sbatch scripts/eval_full_testset.sh <ckpt_path> <results_path>` (the #SBATCH lines below are
# a generic starting point; add --partition/--qos if your cluster requires them). Single-GPU
# inference only (devices=1, no DDP), so any GPU works -- no specific model needed.
#
#SBATCH --job-name=stark_eval_testset
#SBATCH --output=logs/stark_eval_testset_%j.out
#SBATCH --error=logs/stark_eval_testset_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1

set -e

: "${DATASET_ROOT:=./data/LibriTTS_R}"
: "${CACHE_ROOT:=./.cache}"

export PATH="$HOME/.local/bin:$PATH"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# Keep the UTMOSv2 pretrained-model download, HF/torch cache, and uv's own package cache
# (downloaded/extracted by `uv sync --extra eval`, e.g. onnxruntime-gpu) under one place instead
# of scattering into $HOME's default cache locations. UTMOSv2 does NOT honor XDG_CACHE_HOME --
# it hardcodes ~/.cache/utmosv2 unless its own (oddly-misspelled) UTMOSV2_CHACHE env var is set.
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export UTMOSV2_CHACHE="$CACHE_ROOT/utmosv2"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$UTMOSV2_CHACHE" "$UV_CACHE_DIR"

CKPT_PATH="${1:?usage: eval_full_testset.sh <ckpt_path> <results_path>}"
RESULTS_PATH="${2:?usage: eval_full_testset.sh <ckpt_path> <results_path>}"

# Snapshot the checkpoint before evaluating rather than reading it live, in case $CKPT_PATH
# belongs to a still-running training job: Lightning's ModelCheckpoint doesn't write via an
# atomic rename, so a read mid-write could load a truncated/corrupt file. Uses SLURM's per-job
# scratch dir if available, else a local tmp dir.
SNAPSHOT_DIR="${SLURM_TMPDIR:-$REPO_ROOT/.tmp}"
mkdir -p "$SNAPSHOT_DIR"
SNAPSHOT_PATH="$SNAPSHOT_DIR/eval_ckpt_snapshot_$$.ckpt"
cp "$CKPT_PATH" "$SNAPSHOT_PATH"
echo "=== snapshotted $CKPT_PATH -> $SNAPSHOT_PATH ==="

echo "=== installing eval extras (utmosv2) ==="
uv sync --extra eval

echo "=== evaluating $SNAPSHOT_PATH (snapshot of $CKPT_PATH) on test-clean, writing to $RESULTS_PATH ==="
uv run --extra eval scripts/eval_full_testset.py \
    --checkpoint "$SNAPSHOT_PATH" \
    --dataset_root "$DATASET_ROOT" \
    --num_workers 4 \
    --results_path "$RESULTS_PATH"
exit_code=$?

rm -f "$SNAPSHOT_PATH"
echo "=== eval script exited with code $exit_code ==="
exit $exit_code
