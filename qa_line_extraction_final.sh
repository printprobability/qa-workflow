#!/bin/bash
#SBATCH -c 2
#SBATCH --mem-per-cpu 1999MB
#SBATCH -t 48:00:00
#SBATCH -p RM-shared
#SBATCH -o ./logs/qa_le_slurm-%j.out

# Author: Jonathan Armoza
# Creation Date: October 31, 2023
# Script Info:
# Runs line extraction QA script on given book directory, mimicking run_workflow1_watershed.sh and line_extract_dhsegment.sh

echo "In qa_line_extraction_final.sh"

# Make sure at least a book directory and run ID have been passed to this script
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  fail "qa_line_extraction.sh must be supplied with a line extraction type, book directory, and unique run ID."
fi

# Determine which conda environment to activate
if [ "watershed" == "$1" ]; then
  environment_directory="/ocean/projects/hum160002p/nikolaiv/miniconda3/envs/dh_segment"
else # [ "eynollah" == $1 ]; then
  environment_directory="/ocean/projects/hum160002p/nikolaiv/miniconda3/envs/eynollah"
fi

# Show QA line extraction start time
date

# 1. Activate the conda environment for line extraction
echo "Loading conda environment for line extraction..."
source ~/.bashrc
module load anaconda3
conda init
conda activate $environment_directory

# 2. Run QA for line extraction over this book directory
line_extraction_type=$1
book_directory=$2
run_uuid=$3
python3 -u qa_line_extraction.py $line_extraction_type $book_directory $run_uuid

# Show QA line extraction end time
date

# 3. Deactivate the conda environment for line extraction
conda deactivate

# 4. If line_extract_dhsegment.sh (and its processes) complete successfully, continue
if [ $? -eq 0 ]; then
  echo "Completed line extraction."
else
  echo "Failed line extraction."
fi