#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --account=dev-team
#SBATCH --time=0:30:00          # Maximum runtime
#SBATCH --gres gpu:2
#SBATCH --output=/shared/dcli/lshu/BIRD/OmniSQL/train_and_evaluate/logs/multiturn_eval_stdout.log
#SBATCH --error=/shared/dcli/lshu/BIRD/OmniSQL/train_and_evaluate/logs/multiturn_eval_stderr.log

# Load Conda correctly
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust path if needed 

conda activate omnisql_eval
cd /shared/dcli/lshu/BIRD/OmniSQL/train_and_evaluate
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python eval_open_source_models.py
