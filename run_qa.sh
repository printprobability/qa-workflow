#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=1999mb
#SBATCH -t 12:00:00
#SBATCH -p RM-shared
#SBATCH -o ./logs/qa_slurm-%j.out

if [ -z "$1" ] || [ -z "$2" ]
then
    exit "run_qa.sh must be supplied with a qa module name and a config file."
fi

qa_module=$1
config_filename=$2

python3 -u qa.py $qa_module --config_file $config_filename

# Collate results from the QA run
# if [[ -v $QA_AUTOCROP_COLLATE_REQ ]]
# then
#   python3 -u qa.py $1 --collate --config_file $2 --run_uuid $QA_AUTOCROP_UUID
# fi