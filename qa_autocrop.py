# Author:
# Created:
# Purpose: Tests the autocropper on the test set of books created by create_autocrop_test_dir.py.

# Imports

# Built-ins
import csv
import glob
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Third party
import numpy as np
from PIL import Image

# Custom
from prepare_alignment_input_csv import *
from qa_constants import *
from qa_utilities import *


# Globals

# Constants
AUTOCROP_SCRIPT_LOCATION = "..{0}auto_crop.py".format(os.sep)
CROPTYPE_THRESHOLD_BY_INSIDE = "threshold_by_inside"
CROPTYPE_NON_THRESHOLD_BY_INSIDE = "non_threshold_by_inside"

# sbatch parameters
SBATCH_MEMORY_PER_CPU = "1999mb"
SBATCH_NUMBER_CPUS = "2"
SBATCH_PARTITION = "RM-shared"
SBATCH_TIME = "06:00:00"


# Classes

class QA_Autocrop(QA_Module):

    def __init__(self, p_config):

        self.config = p_config
        self.slurm_job_results = []

    def archive_logs(self):

        # 0. Make an 'archive' folder in the output directory if it does not exist
        if not os.path.exists(self.config[OUTPUT_DIRECTORY] + ARCHIVE_DIRECTORY):
            os.makedirs(self.config[OUTPUT_DIRECTORY] + ARCHIVE_DIRECTORY)

        # 1. Move each item (that's not the archive folder or master log for this run) into the archive folder
        for item in get_items_in_dir(self.config[OUTPUT_DIRECTORY], ["directories", "files"]):
            if ARCHIVE_DIRECTORY != item and MASTER_LOG_FILENAME_PREFIX not in item:
                os.rename(self.config[OUTPUT_DIRECTORY] + item,
                    "{0}archive{1}{2}".format(self.config[OUTPUT_DIRECTORY], os.sep, item))

    def clear_logs(self):

        # Clear all log files in the log folder
        log_filepaths = glob.glob(self.config[OUTPUT_DIRECTORY] + "*.out")
        for filepath in log_filepaths:
            os.unlink(filepath)

    def clear_results(self):

        if RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            for book_directory in get_items_in_dir(format_path(self.config[BOOK_DIRECTORY]), ["directories"]):
                full_bookpath = format_path(self.config[BOOK_DIRECTORY] + book_directory)
                if os.path.exists(full_bookpath + RESULTS_DIRECTORY):
                    shutil.rmtree(full_bookpath + RESULTS_DIRECTORY)
        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            if os.path.exists(self.config[BOOK_DIRECTORY] + RESULTS_DIRECTORY):
                shutil.rmtree(self.config[BOOK_DIRECTORY] + RESULTS_DIRECTORY)

    def collate_results(self):

        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            self.__collate_results_on_book(self.config[BOOK_DIRECTORY] + RESULTS_DIRECTORY + os.sep)
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            # 1. Created merged autocrop results for each book in the book directory
            for book_directory in get_items_in_dir(format_path(self.config[BOOK_DIRECTORY]), ["directories"]):

                full_bookpath = format_path(self.config[BOOK_DIRECTORY] + book_directory)

                # Skip the QA log output directory if it exists in the book directory
                if Path(self.config[OUTPUT_DIRECTORY]).name == Path(full_bookpath):
                    continue
                
                # Write out a merged results file for this book in its results directory
                try:
                    self.__collate_results_on_book(full_bookpath + RESULTS_DIRECTORY + os.sep)
                except:
                    print("Collation of results for book {0} has failed.".format(book_directory))

            # 2. Create one CSV file for all books in qa output directory
            print("Merging all collated autocrop results")
            self.__collate_all_book_results()

    def __collate_all_book_results(self):

        with open(self.config[OUTPUT_DIRECTORY] + "{0}_{1}.csv".format(MERGED_RESULTS_FILENAME_PREFIX, datetime.now().timestamp()), "w") as output_file:
            
            # 1. Read in collated results for each book and write them to the merged file
            header_written = False
            for book_directory in get_items_in_dir(format_path(self.config[BOOK_DIRECTORY]), ["directories"]):

                # A. Get the latest collated csv file
                results_directory = format_path(self.config[BOOK_DIRECTORY] + book_directory + os.sep + RESULTS_DIRECTORY)
                csv_filepaths = []
                for filepath in glob.glob(results_directory + "merged_*.csv"):
                    csv_filepaths.append((filepath, os.path.getctime(filepath)))
                if 0 == len(csv_filepaths):
                    print("No collated csv file found for {0}.".format(book_directory))
                    continue
                sorted_csv_filepaths = sorted(csv_filepaths, key=lambda filepath: filepath[1], reverse=True)
                latest_merged_filepath = sorted_csv_filepaths[0][0]

                # B. Save the csv file contents to the merged file
                with open(latest_merged_filepath, "r") as input_file:
                    
                    # Add the lines from this collated csv file (skipping the header if already written) to the all results csv file
                    output_file.writelines(input_file.readlines()[1:] if header_written else input_file.readlines())
                    header_written = True

    def __collate_results_on_book(self, p_results_directory):

        # 1. Get two most recent csv files for autocropping in the results directory (ignoring other collation csvs)
        csv_filepaths = [(filepath, os.path.getctime(filepath)) \
            for filepath in glob.glob(p_results_directory + "*.csv") if "merged_" not in filepath]
        if len(csv_filepaths) < 2:
            raise Exception("Less than two csv files in the results directory: {0}".format(p_results_directory))
        sorted_csv_filepaths = sorted(csv_filepaths, key=lambda filepath: filepath[1], reverse=True)
        results_filepath1, results_filepath2 = sorted_csv_filepaths[0][0], sorted_csv_filepaths[1][0]

        # 2. Merge results file rows
        with open(results_filepath1, "r") as results_file1:
            
            # A. Save rows from the first results file
            csv_reader1 = csv.DictReader(results_file1)
            results1_rows = []
            for row in csv_reader1:
                results1_rows.append({ key: row[key] for key in csv_reader1.fieldnames })

            # B. Merge rows from the second results file with the first
            with open(results_filepath2, "r") as results_file2:

                # I. Grab second results file row
                csv_reader2 = csv.DictReader(results_file2)
                if csv_reader1.fieldnames != csv_reader2.fieldnames:
                    raise Exception("Csv files being merged don't have same columns in the results directory: {0}".format(p_results_directory))
                results2_rows = []
                for row in csv_reader2:
                    results2_rows.append({ key: row[key] for key in csv_reader2.fieldnames })

                # II. Merge results file rows
                merged_results = results1_rows.copy() + [row for row in results2_rows if "original" != row["autocrop_type"]] 

        # 3. Write results into one csv file in the results directory
        print("Writing merged results for results dir: {0}".format(p_results_directory))
        with open(p_results_directory + "merged_results_{0}.csv".format(datetime.now().timestamp()), "w") as output_file:
            csv_writer = csv.writer(output_file)

            csv_writer.writerow([
                "book_name",
                "total_page_count",
                "autocrop_type",
                "image_name",
                "image_width",
                "image_height",
                "min_pct_dimension_difference",
                "image_area",
                "area_diff_from_original",
                "percent_area_diff_from_original",
                "frobenius_norm_from_original"
            ])

            for row in merged_results:
                csv_writer.writerow([
                    row["book_name"],
                    row["total_page_count"],
                    row["autocrop_type"],
                    row["image_name"],
                    row["image_width"],
                    row["image_height"],
                    row["min_pct_dimension_difference"],
                    row["image_area"],
                    row["area_diff_from_original"],
                    row["percent_area_diff_from_original"],
                    row["frobenius_norm_from_original"]
                ])

    def run(self):

        self.slurm_job_results = []
        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            self.slurm_job_results = self.__run_on_book(self.config[BOOK_DIRECTORY])
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            self.slurm_job_results = self.__run_on_all_books()

        print("Slurm Job Results")
        for index in range(len(self.slurm_job_results)):
            print("Result {0}: {1}".format(index, self.slurm_job_results[index]))

    def __run_on_all_books(self):

        return [ self.__run_on_book(format_path(self.config[BOOK_DIRECTORY] + book_name)) \
            for book_name in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]) \
            if Path(self.config[OUTPUT_DIRECTORY]).name != book_name ]

    def __run_on_book(self, p_book_directory): 

        # 1. Start a process to test autocropping methods on this book
        book_name = Path(p_book_directory).name
        print("Creating slurm job for QA of cropping " + book_name, flush=True)

        # A. Build subprocess arguments for sbatch call
        sbatch_args = {

            "-c": SBATCH_NUMBER_CPUS,
            "--mem-per-cpu": SBATCH_MEMORY_PER_CPU,
            "-t": SBATCH_TIME,
            "-o": "{0}slurm-{1}-{2}.out".format(self.config[OUTPUT_DIRECTORY], book_name, datetime.now().timestamp()),
            "-p": SBATCH_PARTITION
        }
        subprocess_cmd = "sbatch"
        for arg in sbatch_args:
            subprocess_cmd += " {0} {1}".format(arg, sbatch_args[arg])
        subprocess_cmd += " {0}{1}qa_autocrop.sh {2}".format(os.getcwd(), os.sep, p_book_directory)

        print("subprocess.run({0}, capture_output=True, text=True, shell=True)".format(subprocess_cmd), flush=True)

        return [subprocess.run(subprocess_cmd, capture_output=True, text=True, shell=True)]


# Main script functions

def output_stats(args):

    # 0. Output path
    output_folder = format_path(args.book_directory)

    csv_results = {}

    book_dir = args.book_directory
    book_name = os.path.basename(book_dir[0:len(book_dir)-1])
    csv_results[book_name] = { "original": {} }
    results_folder = "{0}results{1}".format(output_folder, os.sep)

    print("Book dir: {0}".format(book_dir))
    print("Book name: {0}".format(book_name))

    # 1. Determine info about the original images

    # A. The number of original images
    csv_results[book_name]["original"]["file_count"] = len(get_items_in_dir(str(book_dir), ["files"]))
    csv_results[book_name]["original"]["images"] = {}

    # B. Gather stats on the original book images
    for image_filepath in Path(args.book_directory).glob("*.tif"):

        img = Image.open(image_filepath)
        image_name = os.path.basename(image_filepath)
        csv_results[book_name]["original"]["images"][image_name] = { "binarized_image": binarize_img(img)[0] }

        # Image area
        csv_results[book_name]["original"]["images"][image_name]["image_width"] = img.size[0]
        csv_results[book_name]["original"]["images"][image_name]["image_height"] = img.size[1]
        csv_results[book_name]["original"]["images"][image_name]["image_area"]  = img.size[0] * img.size[1]

        # N/A values
        csv_results[book_name]["original"]["images"][image_name]["area_diff_from_original"] = 0
        csv_results[book_name]["original"]["images"][image_name]["percent_area_diff_from_original"] = 0
        csv_results[book_name]["original"]["images"][image_name]["frobenius_norm_from_original"] = 0
        csv_results[book_name]["original"]["images"][image_name]["min_pct_dimension_difference"] = 0

    # 2. Comparisons between originals and autocrop run
    # for autocrop_type in autocrop_types:
    autocrop_type = CROPTYPE_THRESHOLD_BY_INSIDE if args.threshold_by_inside else CROPTYPE_NON_THRESHOLD_BY_INSIDE

    autocrop_type_subfolder = results_folder + autocrop_type
    # Check to make sure that the cropping run produced a directory of images
    if not os.path.exists(autocrop_type_subfolder):
        print("Cropping for {0} using method '{1}' did not produce any images.".format(book_name, autocrop_type))
        print("No stats csv file for this cropping run will be output.")
        return

    csv_results[book_name][autocrop_type] = {}

    # A. The number of autocropped images
    csv_results[book_name][autocrop_type]["file_count"] = len(get_items_in_dir(autocrop_type_subfolder, ["files"]))
    csv_results[book_name][autocrop_type]["images"] = {} 

    # B. Gather stats on autocropped images and compare to original images
    for image_filepath in Path(autocrop_type_subfolder).glob("*.tif"):

        img = Image.open(image_filepath)
        image_name = os.path.basename(image_filepath)

        # I. Find second to last dash in cropped image filepath
        # original_image_name = image_name[image_name.rfind("-", 0, image_name.rfind("-")) + 1:]
        csv_results[book_name][autocrop_type]["images"][image_name] = {}

        # II. Image area comparison
        csv_results[book_name][autocrop_type]["images"][image_name]["image_width"] = img.size[0]
        csv_results[book_name][autocrop_type]["images"][image_name]["image_height"] = img.size[1]

        # min( (width - width_original) / width_original, (height - height_original) / height_original) )
        autocropped_width = csv_results[book_name][autocrop_type]["images"][image_name]["image_width"]
        autocropped_height = csv_results[book_name][autocrop_type]["images"][image_name]["image_height"]
        original_width = csv_results[book_name]["original"]["images"][image_name]["image_width"]
        original_height = csv_results[book_name]["original"]["images"][image_name]["image_height"]
        csv_results[book_name]["original"]["images"][image_name]["min_pct_dimension_difference"] = \
            min((autocropped_width - original_width) / original_width,
                (autocropped_height - original_height) / original_height)

        csv_results[book_name][autocrop_type]["images"][image_name]["image_area"]  = img.size[0] * img.size[1]
        csv_results[book_name][autocrop_type]["images"][image_name]["area_diff_from_original"] = \
            csv_results[book_name]["original"]["images"][image_name]["image_area"] - \
            csv_results[book_name][autocrop_type]["images"][image_name]["image_area"]
        csv_results[book_name][autocrop_type]["images"][image_name]["percent_area_diff_from_original"] = 100.0 * \
            (float(csv_results[book_name][autocrop_type]["images"][image_name]["image_area"]) / \
             float(csv_results[book_name]["original"]["images"][image_name]["image_area"]))
        
        # III. Frobenius norm between original and autocropped images

        # a. Pad the autocropped image to the size of the original
        new_image = Image.new(
            img.mode,
            (csv_results[book_name]["original"]["images"][image_name]["image_width"],
            csv_results[book_name]["original"]["images"][image_name]["image_height"]),
        ) 
        new_image.paste(img, (0, 0))

        # b. Binarize the autocropped image
        autocrop_img_mtx = np.asarray(binarize_img(new_image)[0]).astype(int)

        # c. Calculate the Frobenius norm between the two binarized images
        original_img_mtx = np.asarray(csv_results[book_name]["original"]["images"][image_name]["binarized_image"]).astype(int)
        diffed_img_mtx = np.subtract(original_img_mtx, autocrop_img_mtx)
        csv_results[book_name][autocrop_type]["images"][image_name]["frobenius_norm_from_original"] = np.linalg.norm(diffed_img_mtx, "fro")

    # 3. Output a csv file of these stats in the autocrop result folder
    for book_name in csv_results:

        results_folder = "{0}results{1}".format(output_folder, os.sep)
        stats_filepath = results_folder + "autocrop_results_{0}.csv".format(datetime.now().timestamp())

        with open(stats_filepath, "w") as output_file:

            csv_writer = csv.writer(output_file)

            csv_writer.writerow([
                "book_name",
                "total_page_count",
                "autocrop_type",
                "image_name",
                "image_width",
                "image_height",
                "min_pct_dimension_difference",
                "image_area",
                "area_diff_from_original",
                "percent_area_diff_from_original",
                "frobenius_norm_from_original"
            ])

            for autocrop_type in csv_results[book_name]:

                for image_name in csv_results[book_name][autocrop_type]["images"]:

                    csv_writer.writerow([book_name,
                                         csv_results[book_name][autocrop_type]["file_count"],
                                         autocrop_type,
                                         image_name,
                                         csv_results[book_name][autocrop_type]["images"][image_name]["image_width"],
                                         csv_results[book_name][autocrop_type]["images"][image_name]["image_height"],
                                         csv_results[book_name][autocrop_type]["images"][image_name]["min_pct_dimension_difference"],
                                         csv_results[book_name][autocrop_type]["images"][image_name]["image_area"],
                                         csv_results[book_name][autocrop_type]["images"][image_name]["area_diff_from_original"],
                                         csv_results[book_name][autocrop_type]["images"][image_name]["percent_area_diff_from_original"],
                                         csv_results[book_name][autocrop_type]["images"][image_name]["frobenius_norm_from_original"]])

def parse_args():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("book_directory", help="Directory containing the images of one book copied using the create_autocrop_test_dir.py script")
    parser.add_argument("--threshold_by_inside", help="Computes binary threshold off inner chunk of page.", action="store_true")

    args = parser.parse_args()

    return args

def run_autocrop(args):

    """ Call auto_crop.py for the given book """
    
    autocrop_type = CROPTYPE_THRESHOLD_BY_INSIDE if args.threshold_by_inside else CROPTYPE_NON_THRESHOLD_BY_INSIDE
            
    # 1. Determine output path for cropped images and create it if it does not exist
    output_path = "{0}results{1}{2}{1}".format(format_path(str(args.book_directory)), os.sep, autocrop_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 2. Run auto_crop.py on the book with the given arguments
    subprocess_args = [
        "python3",
        AUTOCROP_SCRIPT_LOCATION,
        "--path", str(args.book_directory),
        "--output_path", output_path,
        "--test"]
    if CROPTYPE_THRESHOLD_BY_INSIDE == autocrop_type:
        subprocess_args.append("--threshold_by_inside")
    subprocess_args.append("*.tif")

    print("Running command: {0}".format(" ".join(subprocess_args)))
    subprocess.run(subprocess_args)


if __name__ == "__main__":

    args = parse_args()
    run_autocrop(args)
    output_stats(args)