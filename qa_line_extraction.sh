#!/bin/bash

# Author: Jonathan Armoza
# Creation Date: October 31, 2023
# Script Info:
# Runs line extraction QA script on given book directory

echo "In qa_line_extraction.sh"

# 1. Load the environment
source ~/.bashrc
module load anaconda3
conda init
conda activate "/ocean/projects/hum160002p/gsell/.conda/envs/my_env"

# 2. Make sure at least a book directory and run ID have been passed to this script
if [ -z "$1" ] || [ -z "$2" ]
then
  fail "qa_line_extraction.sh must be supplied with a book directory and unique run ID."
fi

# 3. Run QA for line extraction over this book directory

# 1. Switch to the top level book directory
cd $1

# 2. Run line_extract_dhsegment.dh

# A. Date is shown
date

# B. Enables Max's conda environment at gsell/.conda/envs/my_env
source ~/.bashrc
module load anaconda3
conda init
conda activate "/ocean/projects/hum160002p/gsell/.conda/envs/my_env"

# C. Make subdirectories called 'book' and 'book_color'
echo "Making 'book' subdirectory"
echo "Making 'book_color' subdirectory"
mkdir book book_color

# D. Copy original images to 'book_color' subdirectory
cp *.tif book_color

# E. Move page tifs to 'book' subdirectory
echo "Moving original page tifs to 'book' subdirectory"
mv *.tif book/

# F. Make directory pages
# G. Make directory pages_color
echo "Making 'pages' subdirectory"
echo "Making 'pages_color' subdirectory"
mkdir pages pages_color

# H. Copy all tifs in 'book' subdirectory to 'pages'
echo "Copying page tifs to 'pages' subdirectory"
cp book/*.tif pages/

# I. Copy all tifs in book_color to pages_color
echo "Copying color page tifs to 'pages_color' subdirectory"
cp book_color/*.tif pages_color/

# J. Copy code/run_dhsegment_on_book.py to 'pages'
echo "Copying run_dhsegment_on_book.py to 'pages' subdirectory"
cp /ocean/projects/hum160002p/shared/books/code/run_dhsegment_on_book.py pages/

# K. Deactivate Max's conda environment
conda deactivate
source ~/.bashrc
module unload anaconda3

# L. Activate Nikolai's miniconda environment at nikolaiv/miniconda3/envs/dh_segment
conda init
conda activate /ocean/projects/hum160002p/nikolaiv/miniconda3/envs/dh_segment 

# M. Make lines directory
# N. Make lines_color directory
echo "Making 'lines' subdirectory"
echo "Making 'lines_color' subdirectory"
mkdir lines lines_color

# O. Change directory to pages
echo "Changing to 'pages' subdirectory"
cd pages

# P. Run run_dhsegment_on_book.py (recently copied here in 'pages')
echo "Running run_dhsegment_on_book.py for $(basename $1)"
python3 run_dhsegment_on_book.py

# Q. Move back up one directory (cd ..)
echo "Moving back up top level book directory"
cd ..

# R. Copy code/do_all_lines_dhsegment.py lines
echo "Copying do_all_lines_dhsegment.py to 'lines' subdirectory"
cp /ocean/projects/hum160002p/shared/books/code/do_all_lines_dhsegment.py lines/

# S. Change directory to lines
echo "Changing to the 'lines' directory"
cd lines

# T. Copy code/watershed_line_extraction.py to 'lines' subdirectory (present directory)
echo "Copying watershed_line_extraction.py to 'lines subdirectory"
cp /ocean/projects/hum160002p/shared/books/code/watershed_line_extraction.py .

# U. Run watershed_line_extraction.py
echo "Running watershed_line_extraction.py on pages of $(basename $1)"
python3 -u watershed_line_extraction.py ../pages_color ../pages ../pages/dhsegment_output --lines_output_directory . --color_lines_output_directory ../lines_color --max_height 200 --min_width 200 --extension .tif --line_height_quantile 0.85 --transformations_csv ../lines_color/transformations.csv

echo "Done with watershed line extraction."

# V. Date is shown
date

# W. Move up one directory
echo "Moving up one directory"
cd ..

# X. Copy original images back to top-level book directory?
# cp book/*.tif .

# 3. If line_extract_dhsegment.sh (and its processes) complete successfully, continue
if [ $? -eq 0 ]; then
  echo "Completed line extraction"
else
  echo "Failed line extraction"
fi

# 4. 

#   if [ -z "$3" ]
#   then
#     echo "QAing autocrop for $(basename $1) with non_threshold_by_inside..."
#   else
#     echo "QAing autocrop for $(basename $1) with threshold_by_inside..."
#   fi

#   python3 qa_autocrop.py $1 $2 $3
#   # python3 qa_autocrop.py $1 $2
#   # python3 qa_autocrop.py $1 $2 --threshold_by_inside
# fi




