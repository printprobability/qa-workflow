#!/bin/bash
#SBATCH --ntasks-per-node 1
#SBATCH -t 48:00:00
#SBATCH -p RM-shared
#SBATCH -o ./logs/qa_le_slurm-%j.out

# Author: Jonathan Armoza
# Creation Date: January 12, 2024
# Script Info:
# Runs line extraction QA script on given book directory, calling run_workflow1_eynollah.sh

BOOK_DIRECTORY=$1
CODE_DIRECTORY="/ocean/projects/hum160002p/shared/books/code"
RUN_UUID=$2

sbatch ${CODE_DIRECTORY}/run_workflow_1_eynollah.sh $BOOK_DIRECTORY --qa $RUN_UUID
