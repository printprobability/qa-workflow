# Author: Jonathan Armoza
# Created: October 30, 2023
# Purpose: Tests line extraction on the test set of books.

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
from statistics import median, variance

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
DIRECTORY_PAGES = "pages" + os.sep
DIRECTORY_PAGES_COLOR = "pages_color" + os.sep
DIRECTORY_LINES = "lines" + os.sep
DIRECTORY_LINES_COLOR = "lines_color" + os.sep
LINEEXTRACTION_SCRIPT_LOCATION_DHSEGMENT = QA_CODE_DIRECTORY + "run_dhsegment_on_book.py"
LINEEXTRACTION_SCRIPT_LOCATION_WATERSHED = QA_CODE_DIRECTORY + "watershed_line_extraction.py"

LINEEXTRACTION_TYPE_WATERSHED = "watershed"
LINEEXTRACTION_TYPES = [
    LINEEXTRACTION_TYPE_WATERSHED
]

ERRORS_FILE_PREFIX = "line_extraction_errors"
MASTER_LOG_FILENAME_PREFIX = "qa_le_slurm"
MERGED_RESULTS_FILENAME_PREFIX = "le_all_results_merged"
STATS_FILE_PREFIX = "line_extraction_results"

# sbatch parameters
SBATCH_MEMORY_PER_CPU = "1999mb"
SBATCH_NUMBER_CPUS = "2"
SBATCH_PARTITION = "RM-shared"
SBATCH_TIME = "48:00:00"


# Classes

class QA_LineExtraction(QA_Module):

    def __init__(self, p_config):

        self.config = p_config
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

        pass

    def collate_results(self):

        # NOTE: Once more than one line extraction method is introduced,
        # this method will need to be refactored and __collate_results_on_book
        # will need to be adapted for QA line extraction

        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            # self.__collate_results_on_book(self.config[BOOK_DIRECTORY] + RESULTS_DIRECTORY + os.sep)
            pass
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:

            # # 1. Created merged line extraction results for each book in the book directory
            # for book_directory in get_items_in_dir(format_path(self.config[BOOK_DIRECTORY]), ["directories"]):

            #     full_bookpath = format_path(self.config[BOOK_DIRECTORY] + book_directory)

            #     # Skip the QA log output directory if it exists in the book directory
            #     if Path(self.config[OUTPUT_DIRECTORY]).name == Path(full_bookpath):
            #         continue
                
            #     # Write out a merged results file for this book in its results directory
            #     try:
            #         self.__collate_results_on_book(full_bookpath + RESULTS_DIRECTORY + os.sep)
            #     except Exception as e:
            #         print("Collation of results for book {0} has failed.".format(book_directory))
            #         traceback.print_exc(file=sys.stdout)

            # 2. Create one CSV file for all books in qa output directory
            print("Merging all collated autocrop results")
            self.__collate_all_book_results()

    def __collate_all_book_results(self):

        with open(self.config[OUTPUT_DIRECTORY] + "{0}_{1}.csv".format(MERGED_RESULTS_FILENAME_PREFIX, self.config[RUN_UUID]), "w") as output_file:
            
            # 1. Read in collated results for each book and write them to the merged file
            header_written = False
            for book_directory in get_items_in_dir(format_path(self.config[BOOK_DIRECTORY]), ["directories"]):

                # A. Get the latest collated csv file
                # results_directory = format_path(self.config[BOOK_DIRECTORY] + book_directory + os.sep + RESULTS_DIRECTORY)
                # csv_filepaths = []
                # for filepath in glob.glob(results_directory + "merged_*.csv"):
                #     csv_filepaths.append((filepath, os.path.getctime(filepath)))
                # if 0 == len(csv_filepaths):
                #     print("No collated csv file found for {0}.".format(book_directory))
                #     continue
                # sorted_csv_filepaths = sorted(csv_filepaths, key=lambda filepath: filepath[1], reverse=True)
                # latest_merged_filepath = sorted_csv_filepaths[0][0]
                results_directory = format_path(self.config[BOOK_DIRECTORY] + book_directory + os.sep + RESULTS_DIRECTORY)
                latest_merged_filepath = results_directory + "{0}_{1}_{2}.csv".format(STATS_FILE_PREFIX, LINEEXTRACTION_TYPE_WATERSHED, self.config[RUN_UUID])

                # B. Save the csv file contents to the merged file
                with open(latest_merged_filepath, "r") as input_file:
                    
                    # Add the lines from this collated csv file (skipping the header if already written) to the all results csv file
                    output_file.writelines(input_file.readlines()[1:] if header_written else input_file.readlines())
                    header_written = True

    def __collate_results_on_book(self, p_results_directory):

        # 1. Get two most recent csv files for line extraction in the results directory (ignoring other collation csvs)
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

            # csv_writer.writerow([
            #     "book_name",
            #     "total_page_count",
            #     "autocrop_type",
            #     "image_name",
            #     "image_width",
            #     "image_height",
            #     "min_pct_dimension_difference",
            #     "image_area",
            #     "area_diff_from_original",
            #     "percent_area_diff_from_original",
            #     "frobenius_norm_from_original",
            #     "error"
            # ])
            csv_writer.writerow([
                # Line extraction results header columns
            ])

            for row in merged_results:
                # csv_writer.writerow([
                #     row["book_name"],
                #     row["total_page_count"],
                #     row["autocrop_type"],
                #     row["image_name"],
                #     row["image_width"],
                #     row["image_height"],
                #     row["min_pct_dimension_difference"],
                #     row["image_area"],
                #     row["area_diff_from_original"],
                #     row["percent_area_diff_from_original"],
                #     row["frobenius_norm_from_original"],
                #     row["error"]
                # ])
                csv_writer.writerow([
                    # row[line extraction rows],...
                ])

    def output_stats(self):

        print("Entering QA_LineExtraction.output_stats")

        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            self.__output_stats_on_book(self.config[BOOK_DIRECTORY])
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            self.__output_stats_on_all_books()

        print("Exiting QA_LineExtraction.output_stats")

    def __output_stats_on_all_books(self):
                
        for book_name in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):
            if RESULTS_DIRECTORY != book_name:
                self.__output_stats_on_book(format_path(self.config[BOOK_DIRECTORY] + book_name))

    def __output_stats_on_book(self, p_book_directory):

        book_dir = p_book_directory
        book_name = Path(book_dir).name
        le_type = LINEEXTRACTION_TYPE_WATERSHED
        csv_results = { book_name: { le_type: {} } }
        results_folder = format_path(book_dir + RESULTS_DIRECTORY)
        lines_color_folder = format_path(book_dir + DIRECTORY_LINES_COLOR)
        line_df_filepath = lines_color_folder + "line_df.csv"
        pages_color_folder = book_dir + DIRECTORY_PAGES_COLOR 

        # 0. Make a folder for the new output stats file
        makedirs(results_folder)

        # Check to make sure that the cropping run produced a directory of images
        # le_type_subfolder = results_folder + le_type
        # if not os.path.exists(le_type_subfolder):
        #     print("Cropping for {0} using method '{1}' did not produce any images.".format(book_name, autocrop_type))
        #     print("No stats csv file for this cropping run will be output.")
        #     return    

        print("Book dir: {0}".format(book_dir))
        print("Book name: {0}".format(book_name))

        # 0. Potential error file for this cropping run for this book
        # error_filepath = "{0}{1}error_{2}_{3}_{4}.txt".format(results_folder, os.sep,
        #     book_name, le_type, self.config[RUN_UUID])
        # error_lookup = {}
        # if os.path.exists(error_filepath):
        #     error_lookup = read_error_file(error_filepath)
        # print("FINISHED READING ERROR LOOKUP TABLE")
        # print("ERROR_LOOKUP:\n{0}".format(error_lookup))

        # 1. Determine info about the lines extracted for the book pages

        csv_results[book_name][le_type]["images"] = {}
        with open(line_df_filepath, "r") as line_df_file:
            csv_reader = csv.DictReader(line_df_file)

            # A. Store information for each line on each page
            for line_row in csv_reader:

                file_name_parts = line_row["file_name"].split("_page1r_")
                image_name = file_name_parts[0]
                line_number = file_name_parts[1]

                if image_name not in csv_results[book_name][le_type]["images"]:
                    csv_results[book_name][le_type]["images"][image_name] = { "lines": {} }
                csv_results[book_name][le_type]["images"][image_name]["lines"][line_number] = {
                    "width": line_row["width"],
                    "height": line_row["height"]
                }

            # print("CSV RESULTS SO FAR")
            # print(csv_results)
            
            # B. Calculate line metrics for page
            for image_name in csv_results[book_name][le_type]["images"]:

                # I. Number of lines on page
                csv_results[book_name][le_type]["images"][image_name]["num_lines"] = len(csv_results[book_name][le_type]["images"][image_name]["lines"].keys())

                # II. Median height of lines on page
                csv_results[book_name][le_type]["images"][image_name]["median_line_height"] = median([
                    float(csv_results[book_name][le_type]["images"][image_name]["lines"][line_number]["height"]) \
                        for line_number in csv_results[book_name][le_type]["images"][image_name]["lines"]
                ])

                # III. Variance of line height on page
                if len(csv_results[book_name][le_type]["images"][image_name]["lines"].keys()) < 2:
                    csv_results[book_name][le_type]["images"][image_name]["variance_line_height"] = "0"
                else:
                    csv_results[book_name][le_type]["images"][image_name]["variance_line_height"] = variance([
                        float(csv_results[book_name][le_type]["images"][image_name]["lines"][line_number]["height"]) \
                            for line_number in csv_results[book_name][le_type]["images"][image_name]["lines"]
                    ])

                # IV. Page dimensions
                try:
                    img = Image.open(pages_color_folder + image_name + ".tif")
                except UnidentifiedImageError:
                    # error_lookup[Path(image_filepath).name] = str(traceback.format_exc())
                    print(str(traceback.format_exc()))
                    continue
                csv_results[book_name][le_type]["images"][image_name]["image_width"] = img.size[0]
                csv_results[book_name][le_type]["images"][image_name]["image_height"] = img.size[1]
                csv_results[book_name][le_type]["images"][image_name]["image_area"]  = img.size[0] * img.size[1]

                # V. Area of page that is lines
                csv_results[book_name][le_type]["images"][image_name]["area_all_lines"] = sum([
                    (float(csv_results[book_name][le_type]["images"][image_name]["lines"][line_number]["width"]) * \
                    float(csv_results[book_name][le_type]["images"][image_name]["lines"][line_number]["height"])) \
                        for line_number in csv_results[book_name][le_type]["images"][image_name]["lines"]
                ])

        # 2. Add in errored images with their errors
        # for image_name in error_lookup:
        #     print("Adding errored image {0} to csv_results with error {1}".format(image_name, error_lookup[image_name]))
        #     csv_results[book_name][autocrop_type]["images"][image_name] = {}
        #     csv_results[book_name][autocrop_type]["images"][image_name]["image_width"] = "N/A"
        #     csv_results[book_name][autocrop_type]["images"][image_name]["image_height"] = "N/A"
        #     csv_results[book_name][autocrop_type]["images"][image_name]["min_pct_dimension_difference"] = "N/A"
        #     csv_results[book_name][autocrop_type]["images"][image_name]["image_area"] = "N/A"
        #     csv_results[book_name][autocrop_type]["images"][image_name]["area_diff_from_original"] = "N/A"
        #     csv_results[book_name][autocrop_type]["images"][image_name]["percent_area_diff_from_original"] = "N/A"
        #     csv_results[book_name][autocrop_type]["images"][image_name]["frobenius_norm_from_original"] = "N/A"
        #     csv_results[book_name][autocrop_type]["images"][image_name]["error"] = traceback_to_str(error_lookup[image_name])

        # 3. Output a csv file of these stats in the autocrop result folder
        for book_name in csv_results:

            stats_filepath = results_folder + "{0}_{1}_{2}.csv".format(STATS_FILE_PREFIX, le_type, self.config[RUN_UUID])

            with open(stats_filepath, "w") as output_file:

                csv_writer = csv.writer(output_file)

                csv_writer.writerow([
                    "image_name",
                    "line_extraction_type",
                    "image_width",
                    "image_height",                    
                    "image_area",
                    "area_all_lines",
                    "num_lines",
                    "variance_line_height",
                    "median_line_height",
                    "error"
                ])

                for le_type in csv_results[book_name]:

                    for image_name in csv_results[book_name][le_type]["images"]:

                        csv_writer.writerow([
                                            image_name,
                                            le_type,
                                            csv_results[book_name][le_type]["images"][image_name]["image_width"],
                                            csv_results[book_name][le_type]["images"][image_name]["image_height"],
                                            csv_results[book_name][le_type]["images"][image_name]["image_area"],
                                            csv_results[book_name][le_type]["images"][image_name]["area_all_lines"],
                                            csv_results[book_name][le_type]["images"][image_name]["num_lines"],
                                            csv_results[book_name][le_type]["images"][image_name]["variance_line_height"],
                                            csv_results[book_name][le_type]["images"][image_name]["median_line_height"],
                                            ""])
                                            # "'" + csv_results[book_name][le_type]["images"][image_name]["error"] + "'"])
    
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

        print("In __run_on_book")

        # 1. Start a process to test line extraction methods on this book
        book_name = Path(p_book_directory).name

        # 2. Path for error output will be in the top level results directory
        error_path = "{0}results{1}".format(format_path(str(p_book_directory)), os.sep)

        # 3. Spin up a slurm job to extract lines with each possible line extraction type on this book
        slurm_results = []
        le_job_ids = []
        for le_type in LINEEXTRACTION_TYPES:

            # A. Determine output path for line extraction files and create it if it does not exist
            output_path = "{0}results{1}{2}{1}".format(format_path(str(p_book_directory)), os.sep, le_type)
            if not os.path.exists(output_path):
                os.makedirs(output_path)             
        
            print("Creating slurm job for QA of line extraction {0} with line extraction type {1}".format(book_name, le_type))

            # B. Run line extraction on the book with the given arguments

            # I. Save the job ID for outputting stats on line extraction runs afterward
            job_id = "{0}_{1}_{2}".format(book_name, le_type, self.config[RUN_UUID])
            le_job_ids.append(job_id)            

            # II. sbatch arguments
            sbatch_directives = {

                "-c": SBATCH_NUMBER_CPUS,
                "-J": job_id,
                "--mem-per-cpu": SBATCH_MEMORY_PER_CPU,
                "-o": "{0}slurm-{1}_{2}_{3}.out".format(self.config[OUTPUT_DIRECTORY], book_name, le_type, self.config[RUN_UUID]),
                "-p": SBATCH_PARTITION,                
                "-t": SBATCH_TIME
            }

            # III. Build the sbatch call
            subprocess_cmd = "sbatch"
            for arg in sbatch_directives:
                subprocess_cmd += " {0} {1}".format(arg, sbatch_directives[arg])
            subprocess_cmd += " {0}{1}qa_line_extraction_final.sh {2} {3} {4}".format(os.getcwd(), os.sep, p_book_directory, self.config[OUTPUT_DIRECTORY], self.config[RUN_UUID])

            # print("subprocess.run({0}, capture_output=True, text=True, shell=True)".format(subprocess_cmd))
            # slurm_results.append(subprocess.run(subprocess_cmd, capture_output=True, text=True, shell=True))

            # V. Run sbatch and save results
            # slurm_results.append(subprocess.run(subprocess_cmd, capture_output=True, text=True, shell=True))
            slurm_results.append(
                subprocess.Popen(
                    subprocess_cmd,
                    shell=True,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True
                )
            )            
        
        # 4. Build and run a dependent sbatch call for the output of stats on these line extraction runs for this book
        # sbatch_output_stats = "sbatch"
        # sbatch_output_stats += " --dependency=afterok:{0}".format(",".join(le_job_ids))
        # sbatch_output_stats += " qa_autocrop_outputstats.sh"
        # # sbatch_output_stats += " --wrap=\"python3 qa.py line_extraction --single_book --output_stats --book_directory {0} --run_uuid {1}\"".format(book_name, self.config[RUN_UUID])
        # slurm_results.append(
        #     subprocess.Popen(
        #         sbatch_output_stats,
        #         shell=True,
        #         stderr=subprocess.PIPE,
        #         stdout=subprocess.PIPE,
        #         text=True
        #     )
        # )
        # print("subprocess.Popen({0} shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)".format(sbatch_output_stats))   
        
        return slurm_results


# Main script functions

def parse_args():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("book_directory", help="Directory containing the images of one book copied using the create_autocrop_test_dir.py script")
    parser.add_argument("output_directory", help="Output directory where logs for line extraction QA will go")
    parser.add_argument("run_uuid", help="Unique ID for this autocrop run/batch of autocrop runs")
    
    # Optional args here
    # parser.add_argument("--threshold_by_inside", help="Computes binary threshold off inner chunk of page.", action="store_true")

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

            if "BEGIN LINE EXTRACTION FAILURE" in error_lines[index]:
                print("Found BEGIN LINE EXTRACTION FAILURE")
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

def run_line_extraction(args):

    book_directory = format_path(args.book_directory)
    output_directory = format_path(args.output_directory)
    book_name = Path(book_directory).name
    le_type = LINEEXTRACTION_TYPE_WATERSHED

    # 0. Determine output path for cropped images and create it if it does not exist
    output_path = format_path(book_directory + RESULTS_DIRECTORY + le_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 0. Path for error output will be in the top level results directory
    error_path = format_path(book_directory + RESULTS_DIRECTORY)

    # 1. Prepare directory for line extraction and its QA
    print("Preparing directory {0} for line extraction QA".format(book_name))
    
    # A. Make subdirectories that will be used by the line extraction scripts
    print("Making necessary subdirectories...")
    makedirs(book_directory + DIRECTORY_PAGES)
    makedirs(book_directory + DIRECTORY_PAGES_COLOR)
    makedirs(book_directory + DIRECTORY_LINES)
    makedirs(book_directory + DIRECTORY_LINES_COLOR)

    # B. Copy the page images to the line extraction subdirectories
    print("Copying original images to subdirectories...")
    for image_filepath in glob.glob(book_directory + "*.tif"):
        shutil.copy(image_filepath, book_directory + DIRECTORY_PAGES)
        shutil.copy(image_filepath, book_directory + DIRECTORY_PAGES_COLOR)

    # C. Copy Python scripts for parts of line extraction to subdirectories
    print("Copying scripts to subdirectories...")
    shutil.copy(LINEEXTRACTION_SCRIPT_LOCATION_DHSEGMENT, book_directory + DIRECTORY_PAGES)
    shutil.copy(LINEEXTRACTION_SCRIPT_LOCATION_WATERSHED, book_directory + DIRECTORY_LINES)

    # 2. Run line extraction on book

    # A. Move to the 'pages' subdirectory
    os.chdir(book_directory + DIRECTORY_PAGES)

    # B. Run run_dhsegment_on_book.py (recently copied here in 'pages')
    print("Running run_dhsegment_on_book.py for {0}...".format(book_name))
    subprocess_args = [
        "python3",
        Path(LINEEXTRACTION_SCRIPT_LOCATION_DHSEGMENT).name
    ]
    print("Running command: {0}".format(" ".join(subprocess_args)))
    subprocess.run(subprocess_args)    

    # C. Move to the 'lines' subdirectory
    os.chdir(book_directory + DIRECTORY_LINES)

    # D. Run watershed_line_extraction.py
    print("Running watershed_line_extraction.py on pages of {0}...".format(book_name))
    subprocess_args = [
        "python3",
        "-u",
        Path(LINEEXTRACTION_SCRIPT_LOCATION_WATERSHED).name,
        ".." + os.sep + DIRECTORY_PAGES_COLOR,
        ".." + os.sep + DIRECTORY_PAGES,
        "..{0}{1}{0}dhsegment_output".format(os.sep, DIRECTORY_PAGES),
        "--lines_output_directory", ".",
        "--color_lines_output_directory", ".." + os.sep + DIRECTORY_LINES_COLOR,
        "--max_height", "200",
        "--min_width", "200",
        "--extension", ".tif",
        "--line_height_quantile", "0.85",
        "--transformations_csv", "..{0}{1}{0}transformations.csv".format(os.sep, DIRECTORY_LINES_COLOR)
    ]
    print("Running command: {0}".format(" ".join(subprocess_args)))
    subprocess.run(subprocess_args)       

    print("Done with watershed line extraction.")

    # 5. Move up one directory to return to top-level book directory
    os.chdir(book_directory)

if __name__ == "__main__":

    args = parse_args()
    run_line_extraction(args)