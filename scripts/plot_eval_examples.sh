#!/bin/bash
# Generates example articulatory-trace comparison plots (predicted vs ground truth) plus a
# DTW/PCC summary bar chart, from a checkpoint + (optionally) an eval_full_testset.py results
# JSON. Much shorter than eval_full_testset.sh (a handful of example utterances, not the full
# split).
#
# Usage: bash scripts/plot_eval_examples.sh <ckpt_path> <results_json_or_NONE> <out_dir>
# Runs identically with or without SLURM -- direct invocation above, or
# `sbatch scripts/plot_eval_examples.sh <ckpt_path> <results_json_or_NONE> <out_dir>` (the
# #SBATCH lines below are a generic starting point; add --partition/--qos if your cluster
# requires them).
#
#SBATCH --job-name=stark_plot_examples
#SBATCH --output=logs/stark_plot_examples_%j.out
#SBATCH --error=logs/stark_plot_examples_%j.err
#SBATCH --time=00:30:00
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

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$UV_CACHE_DIR"

CKPT_PATH="${1:?usage: plot_eval_examples.sh <ckpt_path> <results_json_or_NONE> <out_dir>}"
RESULTS_JSON="${2:?usage: plot_eval_examples.sh <ckpt_path> <results_json_or_NONE> <out_dir>}"
OUT_DIR="${3:?usage: plot_eval_examples.sh <ckpt_path> <results_json_or_NONE> <out_dir>}"

SNAPSHOT_DIR="${SLURM_TMPDIR:-$REPO_ROOT/.tmp}"
mkdir -p "$SNAPSHOT_DIR"
SNAPSHOT_PATH="$SNAPSHOT_DIR/plot_ckpt_snapshot_$$.ckpt"
cp "$CKPT_PATH" "$SNAPSHOT_PATH"
echo "=== snapshotted $CKPT_PATH -> $SNAPSHOT_PATH ==="

# --extra eval, matching eval_full_testset.sh, even though this script doesn't itself import
# utmosv2: both scripts share one .venv, and a plain `uv sync` here could otherwise race a
# concurrently-running eval_full_testset.sh job and desync the extra it needs mid-run.
echo "=== syncing env ==="
uv sync --extra eval

RESULTS_ARG=()
if [ "$RESULTS_JSON" != "NONE" ]; then
    RESULTS_ARG=(--results_json "$RESULTS_JSON")
fi

echo "=== generating plots into $OUT_DIR ==="
uv run scripts/plot_eval_examples.py \
    --checkpoint "$SNAPSHOT_PATH" \
    --dataset_root "$DATASET_ROOT" \
    --n 4 \
    --out_dir "$OUT_DIR" \
    "${RESULTS_ARG[@]}"
exit_code=$?
rm -f "$SNAPSHOT_PATH"
echo "=== plot script exited with code $exit_code ==="
echo "=== files in $OUT_DIR ==="
ls -la "$OUT_DIR"
exit $exit_code
