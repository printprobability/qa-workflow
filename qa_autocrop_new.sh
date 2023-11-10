#!/bin/bash

# Author: Jonathan Armoza
# Creation Date: October 2, 2023
# Script Info:
# Runs autocrop script on given book directory

# 1. Load the environment
source ~/.bashrc
module load anaconda3
conda init
conda activate "/ocean/projects/hum160002p/gsell/.conda/envs/my_env"

echo "In qa_autocrop_new.sh"
echo "with args"
echo "$@"

# 2. QA autocropping
if [[ "$*" != *"--path"* ]] || [[ "$*" != *"--run_uuid"* ]]
then
  echo "qa_autocrop.sh must be supplied with a book directory via --path and unique run ID via --run_uuid."
else

  # A. Run autocrop.py with given arguments for QA/test
  if [[ "$*" != *"--threshold_by_inside"* ]]
  then
    echo "QAing autocrop for $(basename $1) with non_threshold_by_inside..."
  else
    echo "QAing autocrop for $(basename $1) with threshold_by_inside..."
  fi
  python3 -u $@

  # B. Calculate metrics on the cropping runs and output them into csv files for later collation
  echo "Calculating metrics and outputting results for $3..."
  python3 qa.py autocrop --single_book --output_stats --book_directory $3 --run_uuid $9

fi