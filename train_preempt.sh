#!/bin/bash
#SBATCH --job-name=train_articulatory_tts
#SBATCH --output=logs/train/train_articulatory_tts_%j.out
#SBATCH --error=logs/train/train_articulatory_tts_%j.err
#SBATCH --partition=preempt
#SBATCH --time=5-00:00:00 # Max job run time
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@example.com
#SBATCH --requeue

source /home/YOUR_USERNAME/articulatory-tts/.venv/bin/activate
uv run train.py train=train_large model=large_model # FIXME
