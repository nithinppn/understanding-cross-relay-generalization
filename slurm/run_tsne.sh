#!/bin/bash
#SBATCH --job-name=tsne_compact
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/tsne_%j.out
#SBATCH --error=logs/tsne_%j.err

source ~/.bashrc
conda activate relay

cd /home/woody/iwi5/iwi5469h/understanding-cross-relay-generalization

python3 -m scripts.tsne_compact_mlp_pairwise