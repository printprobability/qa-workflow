#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=1g
#SBATCH -t 12:00:00
#SBATCH -p RM-shared
#SBATCH -o ./logs/slurm-%j.out
#
# Author: Jonathan Armoza
# Creation Date: October 9, 2023
# Script Info:
# Creates one slurm job per book for our autocropping QA script

TEST_BOOK_DIR=/ocean/projects/hum160002p/shared/books/test_autocrop/

# Find all book subdirectories of the test directory
book_dirs="$(find $TEST_BOOK_DIR -maxdepth 1 -type d)"

for book_dir in $book_dirs;
do
   # Skip first value from 'find' (the test directory)
   if [ $book_dir = $TEST_BOOK_DIR ]
   then
      continue
   fi

   echo "Creating slurm job for QA of cropping $book_dir"
   sbatch ./test_autocrop.sh $book_dir
   sbatch ./test_autocrop.sh $book_dir --threshold_by_inside
done