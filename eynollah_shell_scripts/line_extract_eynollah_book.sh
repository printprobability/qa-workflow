#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=2G
#SBATCH -t 48:00:00
#SBATCH -p RM-shared

echo "Preparing for Eynollah line extraction on $1 ..."

# 1. Make required subdirectories in the book folder
cd "$1"
mkdir pages pages_color lines lines_color eynollah_output eynollah_output/extracted_images eynollah_output/pagexml

# 2. Copy original book page images to 'pages' and 'pages_color' subdirectories
cp *.tif pages/
cp *.tif pages_color/

# 3. Load the conda environment for eynollah
echo "Loading conda environment..."
source ~/.bashrc
module load anaconda3
conda init
conda activate /ocean/projects/hum160002p/nikolaiv/miniconda3/envs/eynollah

# Show the line extraction start date/time
date

# 4. Run Eynollah line extraction on this book

# A. Load neural net and make raw predictions
echo "Running eynollah..."
CODE_DIRECTORY="/ocean/projects/hum160002p/shared/books/code"
echo eynollah -m $CODE_DIRECTORY/eynollah_model -di pages_color -o eynollah_output -ep -cl -sa eynollah_output -si eynollah_output/extracted_images
eynollah -m $CODE_DIRECTORY/eynollah_model -di pages_color -o eynollah_output -ep -cl -sa eynollah_output -si eynollah_output/extracted_images

# B. Adjust predictions and get rectangular line output
echo "Starting line extraction on eynollah xml output..."
python3 -u $CODE_DIRECTORY/eynollah_line_image_extraction.py eynollah_output line_minarearect_coords.csv pages_color pages --lines_output_directory lines --color_lines_output_directory lines_color  --ext tif

# Show the line extraction end date/time
echo "Done running line extraction."
date

conda deactivate

