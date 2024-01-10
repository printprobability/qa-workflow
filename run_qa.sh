#!/bin/bash
#SBATCH -ntasks-per-node 1
#SBATCH -t 12:00:00
#SBATCH -p RM-shared
#SBATCH -o ./logs/qa_slurm-%j.out

# Starting point for all QA modules

if [ -z "$1" ] || [ -z "$2" ]
then
    exit "run_qa.sh must be supplied with a qa module name and a config file."
fi

qa_module=$1
config_filename=$2

python3 -u qa.py $qa_module --config_file $config_filename