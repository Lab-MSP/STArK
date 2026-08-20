#!/bin/bash
# Pushes a trained checkpoint to the Hugging Face Hub, then loads it back *from the Hub*
# (round-trip, not just reading the local file) and evaluates it on the LibriTTS-R test-clean
# split, reporting the same PCC/DTW metrics as testing.ipynb / the paper's Table 2.
#
# Runs identically with or without SLURM -- `bash scripts/eval_and_push.sh` directly, or
# `sbatch scripts/eval_and_push.sh` (the #SBATCH lines below are a generic starting point; add
# --partition/--qos if your cluster requires them). Needs a Hugging Face token cached first
# (e.g. via `hf auth login`).
#
#SBATCH --job-name=stark_eval_push
#SBATCH --output=logs/stark_eval_push_%j.out
#SBATCH --error=logs/stark_eval_push_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1

set -e

: "${DATASET_ROOT:=./data/LibriTTS_R}"
: "${OUTPUT_ROOT:=./outputs/stark_large_100k}"
: "${REPO_ID:?set REPO_ID to your target HF Hub model repo, e.g. your-username/stark-large}"
: "${HF_HOME:=./.hf_cache}"

export HF_HOME
export PATH="$HOME/.local/bin:$PATH"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

uv run python scripts/eval_and_push.py \
    --ckpt_path "$OUTPUT_ROOT/ckpt/last.ckpt" \
    --repo_id "$REPO_ID" \
    --dataset_root "$DATASET_ROOT" \
    --overrides model=large_model train=train_large \
        train.checkpoint.dirpath="$OUTPUT_ROOT/ckpt" \
        train.logger.save_dir="$OUTPUT_ROOT/log"
