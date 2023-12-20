#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 48:00:00
#SBATCH -p RM-shared

data_directory=$1

cd $1

for book_directory in */ ; do
    sbatch line_extract_eynollah_book.sh "$book_directory"
    echo "$book_directory"
done
