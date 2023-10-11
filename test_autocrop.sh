#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=1g
#SBATCH -t 12:00:00
#SBATCH -p RM-shared
#SBATCH -o ./logs/slurm-%j.out
#
# Author: Jonathan Armoza
# Creation Date: October 2, 2023
# Script Info:
# Runs autocrop QA script on given book directory
# NOTE: All autocropping types are done in one run of the QA script

ANACONDA_ENVIRONMENT=/ocean/projects/hum160002p/gsell/.conda/envs/my_env

source ~/.bashrc
module load anaconda3
conda init
conda activate "$ANACONDA_ENVIRONMENT"

if [ -z "$2" ]
then
  python3 test_autocrop.py $1
else
  python3 test_autocrop.py $1 $2
fi




