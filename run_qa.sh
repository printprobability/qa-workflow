#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=1g
#SBATCH -t 12:00:00
#SBATCH -p RM-shared
#SBATCH -o ./logs/qa_all_slurm-%j.out

python3 qa.py autocrop --config_file qa.yaml