#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 48:00:00
#SBATCH -p RM-shared

date
source ~/.bashrc
module load anaconda3
conda activate /ocean/projects/hum160002p/gsell/.conda/envs/my_env
CODE=/ocean/projects/hum160002p/shared/books/code

mkdir book
mv *.tif book/

# both b/w and color dirs
mkdir pages pages_color
cp book/*.tif pages/
cp book_color/*.tif pages_color/
cp /ocean/projects/hum160002p/shared/books/code/run_dhsegment_on_book.py pages/

# run the second part of page processing
conda deactivate
source ~/.bashrc
module unload anaconda3
conda init
conda activate /ocean/projects/hum160002p/nikolaiv/miniconda3/envs/eynollah
which python3

#Set up for the line extraction
mkdir lines lines_color
# main output dir
mkdir -p eynollah_output/{extracted_images,pagexml}
# run eynollah on all color pages
# save output to eynollah_output dir
# * xml files
# * extracted images
# * plots
echo "Running eynollah..."
date
echo eynollah -m $CODE/eynollah_model -di pages_color -o eynollah_output -ep -cl -sa eynollah_output -si eynollah_output/extracted_images
eynollah -m $CODE/eynollah_model -di pages_color -o eynollah_output -ep -cl -sa eynollah_output -si eynollah_output/extracted_images
mv eynollah_output/*.xml eynollah_output/pagexml/
echo "Done running eynollah."

# extract minAreaRect lines from eynollah_output pagexml files
# extracts both b/w and color lines from pages/pages_color directories
# - also, we filter out lines above a certain height/below a certain width
# - and standardize line heights for Ocular
echo "Starting line extraction on eynollah xml output..."
date
# python3 -u watershed_line_extraction.py ../pages_color ../pages ../pages/dhsegment_output --lines_output_directory . --color_lines_output_directory ../lines_color --max_height 200 --min_width 200 --extension .tif --line_height_quantile 0.85 --transformations_csv ../lines_color/transformations.csv
python3 -u $CODE/eynollah_line_image_extraction.py eynollah_output line_minarearect_coords.csv pages_color pages --lines_output_directory lines --color_lines_output_directory lines_color  --ext tif # --max_height 200 --min_width 200 --line_height_quantile 0.85 --transformations_csv ../lines_color/transformations.csv
echo "Done running line extraction."
date
