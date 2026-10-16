#!/bin/bash
#SBATCH --job-name=semcons-minilm
#SBATCH --account=def-missro01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/minilm_gpu_%j.out

set -euo pipefail

cd /home/malj30/workspace/semcons_ids/SemCons_IDS_GitHub_Package

source /project/def-missro01/malj30/venvs/semcons_ids/bin/activate

export HF_HOME=/project/def-missro01/malj30/semcons_ids/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python scripts/test_minilm_gpu.py
