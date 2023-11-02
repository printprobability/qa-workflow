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

function fail {

    printf '%s\n' "$1" >&2  ## Send message to stderr. Exclude >&2 if you don't want it that way.
    exit "${2-1}"  ## Return a code specified by $2 or 1 by default.
}

echo "In qa_line_extraction_streamlined.sh"

# 0. Make sure at least a book directory and run ID have been passed to this script
# if [ -z "$1" ] || [ -z "$2" ]
# then
#   fail "qa_line_extraction.sh must be supplied with a book directory and unique run ID."
# fi
if [ -z "$1" ]
then
  fail "qa_line_extraction.sh must be supplied with a book directory."
fi

# 0. Show QA line extraction start time
date

# 1. Prepare directory for line extraction and its QA
echo "Preparing directory $1 for line extraction QA"
echo "Making necessary subdirectories..."
cd $1
# mkdir book book_color pages pages_color lines lines_color
mkdir pages pages_color lines lines_color
echo "Copying original images to subdirectories..."
# cp *.tif book_color/
cp *.tif pages/
cp *.tif pages_color/
echo "Copying scripts to subdirectories..."
cp /ocean/projects/hum160002p/shared/books/code/run_dhsegment_on_book.py pages/
cp /ocean/projects/hum160002p/shared/books/code/watershed_line_extraction.py lines/

# 2. Activate Nikolai's conda environment at nikolaiv/miniconda3/envs/dh_segment
echo "Loading conda environment for line extraction..."
source ~/.bashrc
module load anaconda3
conda init
conda activate /ocean/projects/hum160002p/nikolaiv/miniconda3/envs/dh_segment 

# 3. Run line extraction on book

# A. Move to 'pages' subdirectory
cd pages

# B. Run run_dhsegment_on_book.py (recently copied here in 'pages')
echo "Running run_dhsegment_on_book.py for $(basename $1)..."
python3 run_dhsegment_on_book.py

# C. Move to 'lines' subdirectory
cd ../lines

# D. Run watershed_line_extraction.py
echo "Running watershed_line_extraction.py on pages of $(basename $1)..."
python3 -u watershed_line_extraction.py ../pages_color ../pages ../pages/dhsegment_output --lines_output_directory . --color_lines_output_directory ../lines_color --max_height 200 --min_width 200 --extension .tif --line_height_quantile 0.85 --transformations_csv ../lines_color/transformations.csv

echo "Done with watershed line extraction."

# 4. Show QA line extraction start time
date

# 5. Move up one directory to return to top-level book directory
cd ..

# 6. Copy original images back to top-level book directory?
# cp book/*.tif .

# 7. Deactivate the conda environment for line extraction
conda deactivate

# 8. If line_extract_dhsegment.sh (and its processes) complete successfully, continue
if [ $? -eq 0 ]; then
  echo "Completed line extraction."
else
  echo "Failed line extraction."
fi

# 9. Run QA for line extraction over this book directory
python3 qa_line_extraction.py $1