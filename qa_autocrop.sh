#!/bin/bash

slurm_log_name="./logs/slurm-$(basename $1)-`date +%F-%H-%M-%S`.out"

# Author: Jonathan Armoza
# Creation Date: October 2, 2023
# Script Info:
# Runs autocrop QA script on given book directory
# NOTE: All autocropping types are done in one run of the QA script

#SBATCH -c 2
#SBATCH --mem-per-cpu 1999MB
#SBATCH -t 06:00:00
#SBATCH -p RM-shared
#SBATCH -o ${slurm_log_name}

# 1. Load the environment
source ~/.bashrc
module load anaconda3
conda init
conda activate "/ocean/projects/hum160002p/gsell/.conda/envs/my_env"

echo "In qa_autocrop.sh"

# 2. Run the QA script
if [ -z "$1" ]
then
  echo "qa_autocrop.sh must be supplied with a book directory."
else
  echo "QAing autocrop for $(basename $1)..."
  python3 qa_autocrop.py $1
  python3 qa_autocrop.py $1 --threshold_by_inside
fi




