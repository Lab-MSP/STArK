#!/bin/bash
# Stages and uploads preprocessed LibriTTS-R SPARC features to a Hugging Face dataset repo.
# Needs a Hugging Face token cached first (e.g. via `hf auth login`).
#
# Runs identically with or without SLURM -- `bash scripts/build_hf_dataset.sh` directly, or
# `sbatch scripts/build_hf_dataset.sh` (the #SBATCH lines below are a generic starting point;
# add --partition/--qos if your cluster requires them). This is I/O-bound, no GPU needed.
#
#SBATCH --job-name=build_hf_dataset
#SBATCH --output=logs/build_hf_dataset_%j.out
#SBATCH --error=logs/build_hf_dataset_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4

set -e

: "${SOURCE_ROOT:=./data/LibriTTS_R}"
: "${STAGING_ROOT:=./outputs/libritts-r-stark-staging}"
: "${REPO_ID:?set REPO_ID to your target HF Hub dataset repo, e.g. your-username/libritts-r-stark}"
: "${HF_HOME:=./.hf_cache}"

export HF_HOME
export PATH="$HOME/.local/bin:$PATH"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

uv run --no-project --with huggingface_hub python scripts/build_hf_dataset.py \
    --source_root "$SOURCE_ROOT" \
    --staging_root "$STAGING_ROOT" \
    --repo_id "$REPO_ID" \
    --upload
