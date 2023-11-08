#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=1g
#SBATCH -t 12:00:00
#SBATCH -p RM-shared
#SBATCH -o ./logs/qa_autocrop_slurm-%j.out

if [ -z "$1" ] || [ -z "$2" ]
then
    exit "run_qa.sh must be supplied with a qa module name and a config file."
fi

python3 -u qa.py $1 --config_file $2