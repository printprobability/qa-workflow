# Author: Jonathan Armoza
# Created: October 30, 2023
# Purpose: Tests line extraction on the test set of books.

# Imports

# Built-ins
import ast
import csv
import glob
import math
import os
import shutil
import subprocess
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
LINEEXTRACTION_WATERSHED_METADATA_FILE = "line_df.csv"
MASTER_LOG_FILENAME_PREFIX = "qa_slurm"
MERGED_RESULTS_FILENAME_PREFIX = "le_all_results_merged"
STATS_FILE_PREFIX = "line_extraction_results"

# Watershed error files
DHSEGMENT_ERROR_FILENAME = "le_dhsegment_errors_*.txt"
WATERSHED_ERROR_FILENAME = "le_watershed_errors_*.txt"
WATERSHED_MERGED_ERROR_FILENAME = "le_watershed_all_errors_{}.txt"

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

    # 'archive' commands

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

    # 'clear' subcommands

    def clear_logs(self):

        # Clear all log files in the log folder
        log_filepaths = glob.glob(self.config[OUTPUT_DIRECTORY] + "*.out")
        for filepath in log_filepaths:
            if MASTER_LOG_FILENAME_PREFIX not in filepath and \
               ".gitignore" not in filepath:
                os.unlink(filepath)
                wait_while_exists(filepath)

    def clear_results(self):

        # Output of watershed line extraction

        if RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            for book_directory in get_items_in_dir(format_path(self.config[BOOK_DIRECTORY]), ["directories"]):
                full_bookpath = format_path(self.config[BOOK_DIRECTORY] + book_directory)
                if os.path.exists(full_bookpath + RESULTS_DIRECTORY):
                    shutil.rmtree(full_bookpath + RESULTS_DIRECTORY, ignore_errors=True)
                    wait_while_exists(full_bookpath + RESULTS_DIRECTORY)
        elif RUN_TYPE_SINGLE == self.config[RUN_TYPE]:

            # Remove 'pages' subdirectory
            if os.path.exists(self.config[BOOK_DIRECTORY] + DIRECTORY_PAGES):
                shutil.rmtree(self.config[BOOK_DIRECTORY] + DIRECTORY_PAGES, ignore_errors=True)
                wait_while_exists(self.config[BOOK_DIRECTORY] + DIRECTORY_PAGES)

            # Remove 'pages_color' subdirectory
            if os.path.exists(self.config[BOOK_DIRECTORY] + DIRECTORY_PAGES_COLOR):
                shutil.rmtree(self.config[BOOK_DIRECTORY] + DIRECTORY_PAGES_COLOR, ignore_errors=True)
                wait_while_exists(self.config[BOOK_DIRECTORY] + DIRECTORY_PAGES_COLOR)

            # Remove 'lines' subdirectory
            if os.path.exists(self.config[BOOK_DIRECTORY] + DIRECTORY_LINES):
                shutil.rmtree(self.config[BOOK_DIRECTORY] + DIRECTORY_LINES, ignore_errors=True)
                wait_while_exists(self.config[BOOK_DIRECTORY] + DIRECTORY_LINES)

            # Remove 'lines_color' subdirectory
            if os.path.exists(self.config[BOOK_DIRECTORY] + DIRECTORY_LINES_COLOR):
                shutil.rmtree(self.config[BOOK_DIRECTORY] + DIRECTORY_LINES_COLOR, ignore_errors=True)
                wait_while_exists(self.config[BOOK_DIRECTORY] + DIRECTORY_LINES_COLOR)


    # 'collate' subcommands and helpers

    def collate(self):

        # For now, only collate errors
        # NOTE: This is a temporary implementation until a new line extraction method
        # is introduced for comparison with the 'watershed' method. Now this should be run
        # *before output_stats, so that errors can be tallied via the combined error files created
        # by __collate_errors_on_book_watershed
        # QA sequence with this implementation is 'clear', 'run', 'collate', 'output_stats'
        self.collate_errors()

    def collate_errors(self):

        for le_type in LINEEXTRACTION_TYPES:

            if LINEEXTRACTION_TYPE_WATERSHED == le_type:

                if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
                    self.__collate_errors_on_book_watershed(self.config[BOOK_DIRECTORY])
                elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
                    for book_directory in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):
                        self.__collate_errors_on_book_watershed(format_path(self.config[BOOK_DIRECTORY] + book_directory))
                    # merge_errors_into_csv()

    def __collate_errors_on_book_watershed(self, p_book_directory):

        # 1. Collect error files from both the run_dhsegment run and the watershed run
        dh_segment_error_filepath = os.path.join(p_book_directory, DIRECTORY_PAGES, DHSEGMENT_ERROR_FILENAME)
        watershed_error_filepath = os.path.join(p_book_directory, DIRECTORY_LINES_COLOR, WATERSHED_ERROR_FILENAME)
        error_dict = {

            "dh_segment": read_error_file(dh_segment_error_filepath, "LE"),
            "watershed": read_error_file(watershed_error_filepath, "LE")
        }

        # 2. Output them into one csv file
        combined_error_filepath = self.config[OUTPUT_DIRECTORY] + WATERSHED_MERGED_ERROR_FILENAME.format(self.config[RUN_UUID])
        with open(combined_error_filepath, "w") as merged_error_file:
            csv_writer = csv.writer(merged_error_file)
            csv_writer.writerow(["error_source", "error"])
            for le_submodule in error_dict:
                for error_source in error_dict[le_submodule]:
                    for error_lines in error_dict[le_submodule][error_source]:
                        csv_writer.writerow([
                            error_source,
                            "\n".join(error_lines)
                        ])

    # NOTE: collate_results and its helpers are under construction/not run until
    # a new line extraction method is introduced for comparison
    def collate_results(self):

        # NOTE: Once more than one line extraction method is introduced,
        # this method will need to be refactored and __collate_results_on_book
        # will need to be adapted for QA line extraction

        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            self.__collate_results_on_book(self.config[BOOK_DIRECTORY] + RESULTS_DIRECTORY + os.sep)
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

                if not os.path.exists(latest_merged_filepath):
                    print("ERROR: Stats file does not exist for {0} at {1}".format(book_directory, latest_merged_filepath))
                    continue 

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

    # 'output_stats' command and helpers

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

        self.__output_stats_about_all_books()

    def __output_stats_about_all_books(self):

        # 0. Stats being tracked across all books
        all_book_stats = {
            le_type: {
                "total_lines": 0,
                "total_errors": 0,
                "median_errors_per_book": 0,
                # Each book also tracks its total lines and total errors
                "books": {}
            } for le_type in LINEEXTRACTION_TYPES
        }

        # 1. Gather stats on all books in the book directory list in the config
        for book_directory in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):

            # For each line extraction type, go through the lines of its main output files (results, errors)
            for le_type in LINEEXTRACTION_TYPES:

                if LINEEXTRACTION_TYPE_WATERSHED == le_type:
                    all_book_stats[book_directory] = self.__gather_stats_on_book(book_directory)

                # Tally this book's stats in overall stats
                all_book_stats["total_lines"] += all_book_stats[book_directory]["total_lines"]
                all_book_stats["total_errors"] += all_book_stats[book_directory]["total_errors"]
                all_book_stats["total_lines"] += all_book_stats[book_directory]["total_lines"]



    def __output_stats_on_book(self, p_book_directory):

        csv_results = { le_type: {} for le_type in LINEEXTRACTION_TYPES }
        # le_type = LINEEXTRACTION_TYPE_WATERSHED
        lines_color_folder = format_path(p_book_directory + DIRECTORY_LINES_COLOR)
        line_df_filepath = lines_color_folder + LINEEXTRACTION_WATERSHED_METADATA_FILE
        pages_color_folder = p_book_directory + DIRECTORY_PAGES_COLOR
        results_folder = format_path(p_book_directory + RESULTS_DIRECTORY)

        # For each line extraction method:
        # 1. Gather and output stats for lines
        # 2. Gather and output stats for book

        for le_type in LINEEXTRACTION_TYPES:

            if LINEEXTRACTION_TYPE_WATERSHED == le_type:

                # 1. Gather and output stats for lines for the watershed line extraction method
                lines_stats_filepath = self.__output_stats_for_lines_watershed(p_book_directory)

                # 2. Gather and output stats for this book given the results of the watershed line extraction method
                if lines_stats_filepath:
                    self.__output_stats_for_book_watershed(p_book_directory, lines_stats_filepath)
                else:
                    __output_error("ERROR: Line stats file could not be created.")
        
        # 0. Make a folder for the new output stats file
        makedirs(results_folder)

        # Check to make sure that the cropping run produced a directory of images
        # le_type_subfolder = results_folder + le_type
        # if not os.path.exists(le_type_subfolder):
        #     print("Cropping for {0} using method '{1}' did not produce any images.".format(book_name, autocrop_type))
        #     print("No stats csv file for this cropping run will be output.")
        #     return    

        print("Book dir: {0}".format(p_book_directory))
        print("Book name: {0}".format(Path(p_book_directory).name))

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

        if not os.path.exists(line_df_filepath):
            print("ERROR: Could not find {0} for {1}".format(LINEEXTRACTION_WATERSHED_METADATA_FILE, Path(p_book_directory).name))
            return

        with open(line_df_filepath, "r") as line_df_file:
            csv_reader = csv.DictReader(line_df_file)

            # A. Store information for each line on each page
            for line_row in csv_reader:

                file_name_parts = line_row["file_name"].split("_page1r_")
                image_name = file_name_parts[0]
                line_number = file_name_parts[1]

                if image_name not in csv_results[book_name][le_type]["images"]:
                    csv_results[book_name][le_type]["images"][image_name] = { "lines": {} }

                try:
                    angle_of_rotation = float(ast.literal_eval(line_row["rect"])[2])
                except:
                    print("ERROR: Problem reading angle of rotation for line: {0}".format(line_row["file_name"]))
                    csv_results[book_name][le_type]["images"][image_name]["lines"][line_number] = {
                        "width": line_row["width"],
                        "height": line_row["height"]
                    }
                    continue

                if math.isclose(angle_of_rotation, 90, abs_tol=1):
                    csv_results[book_name][le_type]["images"][image_name]["lines"][line_number] = {
                        "width": line_row["height"],
                        "height": line_row["width"]
                    }
                elif math.isclose(angle_of_rotation, 00, abs_tol=1):
                    csv_results[book_name][le_type]["images"][image_name]["lines"][line_number] = {
                        "width": line_row["width"],
                        "height": line_row["height"]
                    }
                else:
                    print("ERROR: Angle of rotation, dimension detection error: {0}".format(angle_of_rotation))
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
    
    def __gather_stats_on_book_watershed(self, p_book_directory):

        pass

    # 'run' command and helpers

    def run(self):

        self.slurm_job_results = []

        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            self.slurm_job_results = self.__run_on_book(self.config[BOOK_DIRECTORY])
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            self.slurm_job_results = self.__run_on_all_books()

        print_debug_header("Slurm Job Results")
        for index in range(len(self.slurm_job_results)):
            print("Result {0}: {1}".format(index, self.slurm_job_results[index]))
        print_debug_header()

    def __run_on_all_books(self):

        return [ self.__run_on_book(format_path(self.config[BOOK_DIRECTORY] + book_name)) \
            for book_name in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]) \
            if Path(self.config[OUTPUT_DIRECTORY]).name != book_name ]

    def __run_on_book(self, p_book_directory):

        print("In __run_on_book")

        book_name = Path(p_book_directory).name
        slurm_results = []       
        
        # 1. Start up slurm jobs to test line extraction methods on this book
        for le_type in LINEEXTRACTION_TYPES:
        
            print("Creating slurm job for QA of line extraction {0} with line extraction type {1}".format(book_name, le_type))

            # A. Run line extraction on the book with the given arguments

            # I. sbatch arguments
            sbatch_directives = {

                "-c": SBATCH_NUMBER_CPUS,
                "--mem-per-cpu": SBATCH_MEMORY_PER_CPU,
                "-o": "{0}slurm-{1}_{2}_{3}.out".format(self.config[OUTPUT_DIRECTORY], book_name, le_type, self.config[RUN_UUID]),
                "-p": SBATCH_PARTITION,                
                "-t": SBATCH_TIME
            }

            # II. Build the sbatch call
            subprocess_cmd = "sbatch"
            for arg in sbatch_directives:
                subprocess_cmd += " {0} {1}".format(arg, sbatch_directives[arg])
            subprocess_cmd += " {0}{1}qa_line_extraction_final.sh {2} {3} {4}".format(
                os.getcwd(), os.sep,
                le_type, p_book_directory, self.config[RUN_UUID])

            # III. Run sbatch and save results
            slurm_results.append(
                subprocess.Popen(
                    subprocess_cmd,
                    shell=True,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True
                )
            )            
        
        return slurm_results


# Main script functions

def parse_args():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("line_extraction_type", help="Type of line extraction to be run. Current options: [watershed]")
    parser.add_argument("book_directory", help="Directory containing the images of one book copied using the create_autocrop_test_dir.py script")
    parser.add_argument("run_uuid", help="Unique ID for this autocrop run/batch of autocrop runs")
    
    args = parser.parse_args()

    return args

# def read_error_file(error_filepath):
    
#     error_lookup = {}

#     print("In read_error_file")
#     print("Error filepath: {0}".format(error_filepath))

#     with open(error_filepath, "r") as error_file:
#         error_lines = error_file.readlines()

#         print("Read error lines")

#         begin_error = False
#         image_filename = ""
#         recording_traceback = False
#         tb_lines = []
#         for index in range(len(error_lines)):

#             print("Processing line: {0}".format(error_lines[index]))

#             if "BEGIN LINE EXTRACTION FAILURE" in error_lines[index]:
#                 print("Found BEGIN LINE EXTRACTION FAILURE")
#                 begin_error = True
#                 continue
#             if begin_error and "FILE:" in error_lines[index]:
#                 print("begin_error is true and found FILE")
#                 image_filename = Path(error_lines[index].split("FILE: ")[1].strip()).name
#                 print("image_filename: {0}".format(image_filename))
#                 continue
#             if begin_error and "ERROR:" in error_lines[index]:
#                 print("Found ERROR")
#                 recording_traceback = True
#                 continue
#             if recording_traceback:
#                 if "END" in error_lines[index]:
#                     print("Found error END")
#                     error_lookup[image_filename] = tb_lines.copy()
#                     print("error_lookup[{0}]:\n{1}".format(image_filename, error_lookup[image_filename]))
#                     begin_error = False
#                     image_filename = ""
#                     recording_traceback = False
#                     tb_lines = []
#                 else:
#                     print("Appending error line")
#                     tb_lines.append(error_lines[index])
    
#     return error_lookup

def run_line_extraction_watershed(args):

    book_directory = format_path(args.book_directory)
    book_name = Path(book_directory).name

    # 1. Prepare directory for line extraction and its QA
    print(f"Preparing directory {book_name} for line extraction QA...")
    
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
    print(f"Running run_dhsegment_on_book.py for {book_name}...")
    subprocess_args = [
        "python3",
        Path(LINEEXTRACTION_SCRIPT_LOCATION_DHSEGMENT).name,
        "--test",
        "--run_uuid", args.run_uuid
    ]
    print("Running command: {0}".format(" ".join(subprocess_args)))
    subprocess.run(subprocess_args)    

    # C. Move to the 'lines' subdirectory
    os.chdir(book_directory + DIRECTORY_LINES)

    # D. Run watershed_line_extraction.py
    print(f"Running watershed_line_extraction.py on pages of {book_name}...")
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
        "--transformations_csv", "..{0}{1}{0}transformations.csv".format(os.sep, DIRECTORY_LINES_COLOR),
        "--test",
        "--run_uuid", args.run_uuid
    ]
    print("Running command: {0}".format(" ".join(subprocess_args)))
    subprocess.run(subprocess_args)       

    print("Done with watershed line extraction.")

    # 5. Move up one directory to return to top-level book directory
    os.chdir(book_directory)

if __name__ == "__main__":

    args = parse_args()

    if LINEEXTRACTION_TYPE_WATERSHED == args.line_extraction_type:
        run_line_extraction_watershed(args)