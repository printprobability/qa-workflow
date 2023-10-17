#!/bin/bash
#SBATCH -c 2
#SBATCH --mem-per-cpu 1999MB
#SBATCH -t 06:00:00
#SBATCH -p RM-shared
#SBATCH -o ./logs/slurm-%j.out
#
# Author: Jonathan Armoza
# Creation Date: October 2, 2023
# Script Info:
# Runs autocrop QA script on given book directory
# NOTE: All autocropping types are done in one run of the QA script

# 1. Load the environment
source ~/.bashrc
module load anaconda3
conda init
conda activate "/ocean/projects/hum160002p/gsell/.conda/envs/my_env"

# 2. Run the QA script
if [ -z "$2" ]
then
  python3 qa_autocrop.py $1
else
  python3 qa_autocrop.py $1 $2
fi




