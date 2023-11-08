# Author: Jonathan Armoza
# Created: October 11, 2023
# Purpose: Tests the autocropper on the test set of books created by create_autocrop_test_dir.py.

# Imports

# Built-ins
import csv
import glob
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

# Third party
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

# Custom
from prepare_alignment_input_csv import *
from qa_constants import *
from qa_utilities import *


# Globals

# Constants
AUTOCROP_SCRIPT_LOCATION = "..{0}auto_crop.py".format(os.sep)
CROPTYPE_THRESHOLD_BY_INSIDE = "threshold_by_inside"
CROPTYPE_NON_THRESHOLD_BY_INSIDE = "non_threshold_by_inside"
AUTOCROP_TYPES = [
    CROPTYPE_THRESHOLD_BY_INSIDE,
    CROPTYPE_NON_THRESHOLD_BY_INSIDE
]
ERRORS_FILE_PREFIX = "autocrop_errors"
STATS_FILE_PREFIX = "autocrop_results"

# sbatch parameters
SBATCH_MEMORY_PER_CPU = "1999mb"
SBATCH_NUMBER_CPUS = "2"
SBATCH_PARTITION = "RM-shared"
SBATCH_TIME = "06:00:00"


# Classes

class QA_Autocrop(QA_Module):

    def __init__(self, p_config):

        self.slurm_job_results = []

    def archive_logs(self):

        print("Entering archive_logs")

        # 0. Make an 'archive' folder in the output directory if it does not exist
        if not os.path.exists(self.config[OUTPUT_DIRECTORY] + ARCHIVE_DIRECTORY):
            os.makedirs(self.config[OUTPUT_DIRECTORY] + ARCHIVE_DIRECTORY)

        # 1. Move each item (that's not the archive folder or master log for this run) into the archive folder
        for item in get_items_in_dir(self.config[OUTPUT_DIRECTORY], ["directories", "files"]):
            if ARCHIVE_DIRECTORY != item and MASTER_LOG_FILENAME_PREFIX not in item:
                os.rename(self.config[OUTPUT_DIRECTORY] + item,
                    "{0}archive{1}{2}".format(self.config[OUTPUT_DIRECTORY], os.sep, item))
        
        print("Exiting archive_logs")

    def clear_logs(self):

        # Clear all log files in the log folder
        log_filepaths = glob.glob(self.config[OUTPUT_DIRECTORY] + "*.out")
        for filepath in log_filepaths:
            if MASTER_LOG_FILENAME_PREFIX not in filepath and \
               ".gitignore" not in filepath:
                os.unlink(filepath)
                wait_while_exists(filepath)

    def clear_results(self):

        if RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            for book_directory in get_items_in_dir(format_path(self.config[BOOK_DIRECTORY]), ["directories"]):
                full_bookpath = format_path(self.config[BOOK_DIRECTORY] + book_directory)
                if os.path.exists(full_bookpath + RESULTS_DIRECTORY):
                    shutil.rmtree(full_bookpath + RESULTS_DIRECTORY, ignore_errors=True)
                    wait_while_exists(full_bookpath + RESULTS_DIRECTORY)
        elif RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            if os.path.exists(self.config[BOOK_DIRECTORY] + RESULTS_DIRECTORY):
                shutil.rmtree(self.config[BOOK_DIRECTORY] + RESULTS_DIRECTORY, ignore_errors=True)
                wait_while_exists(self.config[BOOK_DIRECTORY] + RESULTS_DIRECTORY)

    def collate_errors(self):

        with open(self.config[OUTPUT_DIRECTORY] + "{0}_{1}.csv".format(MERGED_RESULTS_FILENAME_PREFIX, self.config[RUN_UUID]), "r") as merged_results_file:

            # 1. Read in errors by individual image in merged results, ignoring N/A's and keeping track of results stats
            errors_by_book = {}
            csv_reader = csv.DictReader(merged_results_file)
            result_count = error_count = 0
            for row in csv_reader:

                if "'N/A'" != row["error"]:

                    book_name = row["book_name"]
                    autocrop_type = row["autocrop_type"]
                    image_name = row["image_name"]
                    full_error = row["error"]                

                    # A. Errors are keyed by the line in their traceback message that contains 'Error:' - common string for Python errors
                    error_type = get_uniquer_error_line(row["error"])

                    # B. Associate image with book and this kind of error
                    if book_name not in errors_by_book:
                        errors_by_book[book_name] = {}
                
                    if error_type not in errors_by_book[book_name]:
                        errors_by_book[book_name][error_type] = []

                    errors_by_book[book_name][error_type].append((image_name, autocrop_type, full_error))

                    error_count += 1
                
                result_count += 1
            
            # 2. Output file with images sorted by book and then their error type, noting autocrop type and the full error as well
            with open("{0}{1}_{2}.csv".format(self.config[OUTPUT_DIRECTORY], ERRORS_FILE_PREFIX, self.config[RUN_UUID]), "w") as output_file:
                csv_writer = csv.writer(output_file)
                csv_writer.writerow([
                    "book_name",
                    "image_name",
                    "autocrop_type",
                    "error_type",
                    "error"
                ])
                for book_name in errors_by_book:
                    for error_type in errors_by_book[book_name]:
                        for index in range(len(errors_by_book[book_name][error_type])):
                            csv_writer.writerow([
                                book_name,
                                errors_by_book[book_name][error_type][index][0],
                                errors_by_book[book_name][error_type][index][1],
                                error_type,
                                errors_by_book[book_name][error_type][index][2]
                            ])

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
                except Exception as e:
                    print("Collation of results for book {0} has failed.".format(book_directory))
                    traceback.print_exc(file=sys.stdout)

            # 2. Create one CSV file for all books in qa output directory
            print("Merging all collated autocrop results")
            self.__collate_all_book_results()

    def __collate_all_book_results(self):

        with open(self.config[OUTPUT_DIRECTORY] + "{0}_{1}.csv".format(MERGED_RESULTS_FILENAME_PREFIX, self.config[RUN_UUID]), "w") as output_file:
            
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
        with open(p_results_directory + "merged_results_{0}.csv".format(self.config[RUN_UUID]), "w") as output_file:
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
                "frobenius_norm_from_original",
                "error"
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
                    row["frobenius_norm_from_original"],
                    row["error"]
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

        slurm_results = []
        for autocrop_type in AUTOCROP_TYPES:
        
            print("Creating slurm job for QA of cropping {0} with autocrop type {1}".format(book_name, autocrop_type))

            # A. Build subprocess arguments for sbatch call
            sbatch_args = {

                "-c": SBATCH_NUMBER_CPUS,
                "--mem-per-cpu": SBATCH_MEMORY_PER_CPU,
                "-t": SBATCH_TIME,
                "-o": "{0}slurm-{1}_{2}_{3}.out".format(self.config[OUTPUT_DIRECTORY], book_name, autocrop_type, self.config[RUN_UUID]),
                "-p": SBATCH_PARTITION
            }
            subprocess_cmd = "sbatch"
            for arg in sbatch_args:
                subprocess_cmd += " {0} {1}".format(arg, sbatch_args[arg])
            subprocess_cmd += " {0}{1}qa_autocrop.sh {2} {3}".format(os.getcwd(), os.sep, p_book_directory, self.config[RUN_UUID])
            if CROPTYPE_THRESHOLD_BY_INSIDE == autocrop_type:
                subprocess_cmd += " --" + autocrop_type

            print("subprocess.run({0}, capture_output=True, text=True, shell=True)".format(subprocess_cmd))

            slurm_results.append(subprocess.run(subprocess_cmd, capture_output=True, text=True, shell=True))
        
        return slurm_results


# Main script functions

def output_stats(args):

    # 0. Output path
    output_folder = format_path(args.book_directory)

    csv_results = {}

    book_dir = args.book_directory
    book_name = os.path.basename(book_dir[0:len(book_dir)-1])
    csv_results[book_name] = { "original": {} }
    results_folder = "{0}results{1}".format(output_folder, os.sep)
    autocrop_type = CROPTYPE_THRESHOLD_BY_INSIDE if args.threshold_by_inside else CROPTYPE_NON_THRESHOLD_BY_INSIDE
    autocrop_type_subfolder = results_folder + autocrop_type

    # Check to make sure that the cropping run produced a directory of images
    # if not os.path.exists(autocrop_type_subfolder):
    #     print("Cropping for {0} using method '{1}' did not produce any images.".format(book_name, autocrop_type))
    #     print("No stats csv file for this cropping run will be output.")
    #     return    

    print("Book dir: {0}".format(book_dir))
    print("Book name: {0}".format(book_name))

    # 0. Potential error file for this cropping run for this book
    error_filepath = "{0}results{1}error_{2}_{3}_{4}.txt".format(output_folder, os.sep,
        book_name, autocrop_type, args.run_uuid)
    error_lookup = {}
    if os.path.exists(error_filepath):
        error_lookup = read_error_file(error_filepath)

    print("FINISHED READING ERROR LOOKUP TABLE")
    print("ERROR_LOOKUP:\n{0}".format(error_lookup))

    # 1. Determine info about the original images

    # A. The number of original images
    csv_results[book_name]["original"]["file_count"] = len(get_items_in_dir(str(book_dir), ["files"]))
    csv_results[book_name]["original"]["images"] = {}

    # B. Gather stats on the original book images
    for image_filepath in Path(args.book_directory).glob("*.tif"):

        try:
            img = Image.open(image_filepath)
        except UnidentifiedImageError:
            error_lookup[Path(image_filepath).name] = str(traceback.format_exc())
            continue
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
        csv_results[book_name]["original"]["images"][image_name]["error"] = "N/A"

    # 2. Comparisons between originals and autocrop run

    csv_results[book_name][autocrop_type] = {}

    # A. The number of autocropped images
    csv_results[book_name][autocrop_type]["file_count"] = len(get_items_in_dir(autocrop_type_subfolder, ["files"]))
    csv_results[book_name][autocrop_type]["images"] = {} 

    # B. Gather stats on autocropped images and compare to original images
    for image_filepath in Path(autocrop_type_subfolder).glob("*.tif"):

        try:
            img = Image.open(image_filepath)
        except Exception as e:
            print("Image opening exception for {0}".format(image_filepath))
            error_lookup[Path(image_filepath).name] = str(traceback.format_exc())
            continue
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
        csv_results[book_name][autocrop_type]["images"][image_name]["min_pct_dimension_difference"] = \
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

        # IV. All images found are likely not errored
        csv_results[book_name][autocrop_type]["images"][image_name]["error"] = "N/A"

    # 3. Add in errored images with their errors
    for image_name in error_lookup:
        print("Adding errored image {0} to csv_results with error {1}".format(image_name, error_lookup[image_name]))
        csv_results[book_name][autocrop_type]["images"][image_name] = {}
        csv_results[book_name][autocrop_type]["images"][image_name]["image_width"] = "N/A"
        csv_results[book_name][autocrop_type]["images"][image_name]["image_height"] = "N/A"
        csv_results[book_name][autocrop_type]["images"][image_name]["min_pct_dimension_difference"] = "N/A"
        csv_results[book_name][autocrop_type]["images"][image_name]["image_area"] = "N/A"
        csv_results[book_name][autocrop_type]["images"][image_name]["area_diff_from_original"] = "N/A"
        csv_results[book_name][autocrop_type]["images"][image_name]["percent_area_diff_from_original"] = "N/A"
        csv_results[book_name][autocrop_type]["images"][image_name]["frobenius_norm_from_original"] = "N/A"
        csv_results[book_name][autocrop_type]["images"][image_name]["error"] = traceback_to_str(error_lookup[image_name])

    # 4. Output a csv file of these stats in the autocrop result folder
    for book_name in csv_results:

        results_folder = "{0}results{1}".format(output_folder, os.sep)
        stats_filepath = results_folder + "{0}_{1}_{2}.csv".format(STATS_FILE_PREFIX, autocrop_type, args.run_uuid)

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
                "frobenius_norm_from_original",
                "error"
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
                                         csv_results[book_name][autocrop_type]["images"][image_name]["frobenius_norm_from_original"],
                                         "'" + csv_results[book_name][autocrop_type]["images"][image_name]["error"] + "'"])

def parse_args():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("book_directory", help="Directory containing the images of one book copied using the create_autocrop_test_dir.py script")
    parser.add_argument("run_uuid", help="Unique ID for this autocrop run/batch of autocrop runs")
    parser.add_argument("--threshold_by_inside", help="Computes binary threshold off inner chunk of page.", action="store_true")

    args = parser.parse_args()

    return args

def read_error_file(error_filepath):
    
    error_lookup = {}

    print("In read_error_file")
    print("Error filepath: {0}".format(error_filepath))

    with open(error_filepath, "r") as error_file:
        error_lines = error_file.readlines()

        print("Read error lines")

        begin_error = False
        image_filename = ""
        recording_traceback = False
        tb_lines = []
        for index in range(len(error_lines)):

            print("Processing line: {0}".format(error_lines[index]))

            if "BEGIN AUTOCROP FAILURE" in error_lines[index]:
                print("Found BEGIN AUTOCROP FAILURE")
                begin_error = True
                continue
            if begin_error and "FILE:" in error_lines[index]:
                print("begin_error is true and found FILE")
                image_filename = Path(error_lines[index].split("FILE: ")[1].strip()).name
                print("image_filename: {0}".format(image_filename))
                continue
            if begin_error and "ERROR:" in error_lines[index]:
                print("Found ERROR")
                recording_traceback = True
                continue
            if recording_traceback:
                if "END" in error_lines[index]:
                    print("Found error END")
                    error_lookup[image_filename] = tb_lines.copy()
                    print("error_lookup[{0}]:\n{1}".format(image_filename, error_lookup[image_filename]))
                    begin_error = False
                    image_filename = ""
                    recording_traceback = False
                    tb_lines = []
                else:
                    print("Appending error line")
                    tb_lines.append(error_lines[index])
    
    return error_lookup

def run_autocrop(args):

    """ Call auto_crop.py for the given book """
    
    autocrop_type = CROPTYPE_THRESHOLD_BY_INSIDE if args.threshold_by_inside else CROPTYPE_NON_THRESHOLD_BY_INSIDE
            
    # 1. Determine output path for cropped images and create it if it does not exist
    output_path = "{0}results{1}{2}{1}".format(format_path(str(args.book_directory)), os.sep, autocrop_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 2. Path for error output will be in the top level results directory
    error_path = "{0}results{1}".format(format_path(str(args.book_directory)), os.sep)

    # 2. Run auto_crop.py on the book with the given arguments
    subprocess_args = [
        "python3",
        AUTOCROP_SCRIPT_LOCATION,
        "--path", str(args.book_directory),
        "--output_path", output_path,
        "--error_path", error_path,
        "--run_uuid", args.run_uuid,
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