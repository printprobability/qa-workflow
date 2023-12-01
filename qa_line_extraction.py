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

import json

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

# Watershed files

# Line extraction output
LINEEXTRACTION_WATERSHED_METADATA_FILE = "line_df.csv"
DHSEGMENT_ERROR_FILENAME = "le_dhsegment_errors_*.txt"
WATERSHED_ERROR_FILENAME = "le_watershed_errors_*.txt"
WATERSHED_MERGED_ERROR_FILENAME = "le_watershed_all_errors_{}.txt"

# QA output
MASTER_LOG_FILENAME_PREFIX = "qa_slurm"
QA_OUTPUT_PREFIX = "le_{}_"
RESULTS_FILENAME_PREFIX = QA_OUTPUT_PREFIX + "results_"
ERRORS_FILENAME_PREFIX = QA_OUTPUT_PREFIX + "errors_"


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

        print("Entering QA_LineExtraction.clear_logs")

        # Clear all log files in the log folder
        for filepath in glob.glob(self.config[OUTPUT_DIRECTORY] + "*.out"):
            if MASTER_LOG_FILENAME_PREFIX not in filepath and \
               ".gitignore" not in filepath:
                os.unlink(filepath)
                wait_while_exists(filepath)

        print("Exiting QA_LineExtraction.clear_logs")

    def clear_results(self):

        print("Entering QA_LineExtraction.clear_results")

        # 0. All subdirectories in line extraction book directory to be deleted
        directories_to_be_removed = [

            DIRECTORY_PAGES,
            DIRECTORY_PAGES_COLOR,
            DIRECTORY_LINES,
            DIRECTORY_LINES_COLOR,
            RESULTS_DIRECTORY,
            "resultswatershed"
        ]

        # 1. Establish list of book directories to look through for subdirectory deletion
        book_directories = None
        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            book_directories = [self.config[BOOK_DIRECTORY]]
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            book_directories = [self.config[BOOK_DIRECTORY] + directory + os.sep for directory in get_items_in_dir(format_path(self.config[BOOK_DIRECTORY]), ["directories"])]

        # 2. Delete listed line extraction subdirectories in all book directories
        for book_directory in book_directories:
            for directory in directories_to_be_removed:
                directory_to_be_removed = book_directory + directory
                if os.path.exists(directory_to_be_removed):
                    shutil.rmtree(directory_to_be_removed, ignore_errors=True)
                    wait_while_exists(directory_to_be_removed)

        print("Exiting QA_LineExtraction.clear_results")


    # 'collate' subcommands and helpers

    def collate(self):

        print("Entering QA_LineExtraction.collate")

        # For now, only collate errors
        # NOTE: This is a temporary implementation until a new line extraction method
        # is introduced for comparison with the 'watershed' method. Now this should be run
        # *before output_stats, so that errors can be tallied via the combined error files created
        # by __collate_errors_on_book_watershed
        # QA sequence with this implementation is 'clear', 'run', 'collate', 'output_stats'
        self.collate_errors()

        print("Exiting QA_LineExtraction.collate")

    def collate_errors(self):

        print("Entering QA_LineExtraction.collate_errors")

        for le_type in LINEEXTRACTION_TYPES:

            if LINEEXTRACTION_TYPE_WATERSHED == le_type:

                if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
                    self.__collate_errors_on_book_watershed(self.config[BOOK_DIRECTORY])
                elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
                    for book_directory in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):
                        self.__collate_errors_on_book_watershed(format_path(self.config[BOOK_DIRECTORY] + book_directory))

        print("Exiting QA_LineExtraction.collate_errors")

    def __collate_errors_on_book_watershed(self, p_book_directory):
    
        print("Entering QA_LineExtraction.__collate_errors_on_book_watershed")

        # 1. Collect error files from both the run_dhsegment run and the watershed run
        dh_segment_error_filepath = os.path.join(p_book_directory, DIRECTORY_PAGES, DHSEGMENT_ERROR_FILENAME)
        watershed_error_filepath = os.path.join(p_book_directory, DIRECTORY_LINES_COLOR, WATERSHED_ERROR_FILENAME)
        error_dict = {

            "dh_segment": read_error_file(dh_segment_error_filepath, "LE"),
            "watershed": read_error_file(watershed_error_filepath, "LE")
        }

        # 2. Create the results directory in the book directory
        results_directory = self.config[BOOK_DIRECTORY] + RESULTS_DIRECTORY + os.sep
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        # 3. Output them into one csv file
        combined_error_filepath = results_directory + WATERSHED_MERGED_ERROR_FILENAME.format(self.config[RUN_UUID])
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

        print("Exiting QA_LineExtraction.__collate_errors_on_book_watershed")

    # NOTE: collate_results and its helpers are under construction/not run until
    # a new line extraction method is introduced for comparison
    def collate_results(self):
    
        print("Entering QA_LineExtraction.collate_results")

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

        print("Exiting QA_LineExtraction.collate_results")

    def __collate_all_book_results(self):
    
        print("Entering QA_LineExtraction.__collate_all_book_results")

        with open(self.config[OUTPUT_DIRECTORY] + "{0}_{1}.csv".format(RESULTS_FILENAME_PREFIX, self.config[RUN_UUID]), "w") as output_file:
            
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

        print("Exiting QA_LineExtraction.__collate_all_book_results")

    def __collate_results_on_book(self, p_results_directory):
    
        print("Entering QA_LineExtraction.__collate_results_on_book")

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
        
        print("Exiting QA_LineExtraction.__collate_results_on_book")


    # 'output_stats' command and helpers

    def output_stats(self):

        print("Entering QA_LineExtraction.output_stats")

        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:

            book_name = Path(self.config[BOOK_DIRECTORY]).name
            booklevel_stats = {
                book_name: self.__output_stats_on_book(self.config[BOOK_DIRECTORY])
            }
            self.__output_stats_booklevel(self.config[BOOK_DIRECTORY], booklevel_stats)
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:

            self.__output_stats_on_all_books()

        print("Exiting QA_LineExtraction.output_stats")

    def __output_stats_on_all_books(self):

        print("Entering QA_LineExtraction.__output_stats_on_all_books")
        
        # 1. Output a results file per book and store booklevel stats that are returned
        booklevel_stats = {}
        for book_name in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):
            if RESULTS_DIRECTORY != book_name:
                booklevel_stats[book_name] = self.__output_stats_on_book(format_path(self.config[BOOK_DIRECTORY] + book_name))

        # 2. Output one file containing booklevel stats of for whole line extraction run
        self.__output_stats_booklevel(booklevel_stats)

        print("Exiting QA_LineExtraction.__output_stats_on_all_books")

    def __output_stats_booklevel(self, p_booklevel_stats):

        print("Entering QA_LineExtraction.__output_stats_booklevel")

        for le_type in LINEEXTRACTION_TYPES:

            results_filename_prefix = RESULTS_FILENAME_PREFIX.format(le_type)
            results_filepath = f"{self.config[OUTPUT_DIRECTORY]}{results_filename_prefix}all_{self.config[RUN_UUID]}.csv"

            with open(results_filepath, "w") as output_file:

                print(f"Writing results to {results_filepath}")

                csv_writer = csv.writer(output_file)

                csv_writer.writerow([
                    "book_name",
                    "pages",
                    "lines",
                    "errors",
                    "unique_errors",
                    "median_page_width",
                    "median_page_height",
                    "median_page_area",
                    "median_all_lines_area",
                    "median_line_height_median",
                    "median_variance_line_height"            
                ])

                for book_name in p_booklevel_stats:

                    csv_writer.writerow([

                        book_name,
                        p_booklevel_stats[book_name][le_type]["total_pages"],
                        p_booklevel_stats[book_name][le_type]["total_lines"],
                        p_booklevel_stats[book_name][le_type]["total_errors"],
                        p_booklevel_stats[book_name][le_type]["total_unique_errors"],
                        p_booklevel_stats[book_name][le_type]["median_image_width"],
                        p_booklevel_stats[book_name][le_type]["median_image_height"],
                        p_booklevel_stats[book_name][le_type]["median_image_area"],
                        p_booklevel_stats[book_name][le_type]["median_area_all_lines"],
                        p_booklevel_stats[book_name][le_type]["median_line_height_median"],
                        p_booklevel_stats[book_name][le_type]["median_variance_line_height"]
                    ])

        print("Exiting QA_LineExtraction.__output_stats_booklevel")

    def __output_stats_on_book(self, p_book_directory):

        print("Entering QA_LineExtraction.__output_stats_on_book")

        csv_results = { le_type: {} for le_type in LINEEXTRACTION_TYPES }

        for le_type in LINEEXTRACTION_TYPES:

            if LINEEXTRACTION_TYPE_WATERSHED == le_type:

                # 1. Gather and output stats for lines for the watershed line extraction method
                csv_results[le_type]["page"] = self.__output_stats_on_book_watershed(p_book_directory)

                # 2. Tally book stats for potential use outside of function
                csv_results[le_type]["book"] = self.__tally_booklevel_stats_watershed(p_book_directory, csv_results[le_type]["page"])

        print("Exiting QA_LineExtraction.__output_stats_on_book")

        # 3. Return just book level stats for each line extraction type
        return { le_type: csv_results[le_type]["book"] for le_type in csv_results }

    def __output_stats_on_book_watershed(self, p_book_directory):

        print("Entering QA_LineExtraction.__output_stats_on_book_watershed")

        csv_results = { "images": {} }
        lines_color_folder = format_path(p_book_directory + DIRECTORY_LINES_COLOR)
        line_df_filepath = lines_color_folder + LINEEXTRACTION_WATERSHED_METADATA_FILE
        pages_color_folder = p_book_directory + DIRECTORY_PAGES_COLOR
        results_folder = format_path(p_book_directory + RESULTS_DIRECTORY)

        # 0. Make a folder for the new output stats file
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # 1. Potential error file for this line extraction run for this book
        error_lookup = {}
        error_filepath = ""
        for filepath in glob.glob(results_folder + WATERSHED_MERGED_ERROR_FILENAME.replace("{}", "*")):
            error_filepath = filepath
            break
        if error_filepath:
            with open(error_filepath, "r") as merged_error_file:
                csv_reader = csv.DictReader(merged_error_file)
                for row in csv_reader:
                    error_lookup[row["error_source"]] = row["error"]

        # 2. Determine info about the lines extracted for the book pages

        if not os.path.exists(line_df_filepath):

            print(f"ERROR: Could not find {LINEEXTRACTION_WATERSHED_METADATA_FILE} for {Path(p_book_directory).name}")
            print("Exiting QA_LineExtraction.__output_stats_on_book_watershed")
            return csv_results
        
        with open(line_df_filepath, "r") as line_df_file:
            csv_reader = csv.DictReader(line_df_file)
            if 0 == len(list(csv_reader)):
                print(f"ERROR: Line extraction output file {LINEEXTRACTION_WATERSHED_METADATA_FILE} is empty.")
                print("Exiting QA_LineExtraction.__output_stats_on_book_watershed")
                return csv_results

        with open(line_df_filepath, "r") as line_df_file:

            csv_reader = csv.DictReader(line_df_file)

            # A. Store information for each line on each page
            for line_row in csv_reader:

                file_name_parts = line_row["file_name"].split("_page1r_")
                image_name = file_name_parts[0]
                line_number = file_name_parts[1]

                if image_name not in csv_results["images"]:
                    csv_results["images"][image_name] = { "lines": {} }

                # A. Save angle of rotation, width, and height
                try:
                    angle_of_rotation = float(ast.literal_eval(line_row["rect"])[2])
                except:
                    print("ERROR: Problem reading angle of rotation for line: {0}".format(line_row["file_name"]))
                    csv_results["images"][image_name]["lines"][line_number] = {
                        "angle": "N/A",
                        "width": line_row["width"],
                        "height": line_row["height"]
                    }
                    continue

                # I. Determine if dimensions need to be flipped due to angle
                if math.isclose(angle_of_rotation, 90, abs_tol=1):

                    csv_results["images"][image_name]["lines"][line_number] = {
                        "angle": angle_of_rotation,
                        "width": line_row["height"],
                        "height": line_row["width"]
                    }
                elif math.isclose(angle_of_rotation, 00, abs_tol=1):

                    csv_results["images"][image_name]["lines"][line_number] = {
                        "angle": angle_of_rotation,
                        "width": line_row["width"],
                        "height": line_row["height"]
                    }
                else:

                    csv_results["images"][image_name]["lines"][line_number] = {
                        "angle": angle_of_rotation,
                        "width": line_row["width"],
                        "height": line_row["height"]
                    }

                # B. Save any errors from error lookup for this line
                csv_results["images"][image_name]["lines"][line_number]["error(s)"] = "N/A"
                if line_row["file_name"] in error_lookup:
                    csv_results["images"][image_name]["lines"][line_number]["error(s)"] = error_lookup[line_row["file_name"]]

            # B. Calculate line metrics for page
            for image_name in csv_results["images"]:

                # I. Number of lines on page
                csv_results["images"][image_name]["num_lines"] = len(csv_results["images"][image_name]["lines"].keys())

                # II. Median height of lines on page
                csv_results["images"][image_name]["median_line_height"] = median([
                    float(csv_results["images"][image_name]["lines"][line_number]["height"]) \
                        for line_number in csv_results["images"][image_name]["lines"]
                ])

                # III. Variance of line height on page
                if len(csv_results["images"][image_name]["lines"].keys()) < 2:
                    csv_results["images"][image_name]["variance_line_height"] = 0.0
                else:
                    csv_results["images"][image_name]["variance_line_height"] = variance([
                        float(csv_results["images"][image_name]["lines"][line_number]["height"]) \
                            for line_number in csv_results["images"][image_name]["lines"]
                    ])

                # IV. Page dimensions
                try:
                    img = Image.open(pages_color_folder + image_name + ".tif")
                except UnidentifiedImageError:
                    print(str(traceback.format_exc()))
                    continue
                csv_results["images"][image_name]["image_width"] = img.size[0]
                csv_results["images"][image_name]["image_height"] = img.size[1]
                csv_results["images"][image_name]["image_area"]  = img.size[0] * img.size[1]

                # V. Area of page that is lines
                areas = [
                    (float(csv_results["images"][image_name]["lines"][line_number]["width"]) * \
                    float(csv_results["images"][image_name]["lines"][line_number]["height"])) \
                        for line_number in csv_results["images"][image_name]["lines"]
                ]
                csv_results["images"][image_name]["area_all_lines"] = sum(areas)

        # 3. Output a csv file of these stats in the line extraction results folder

        stats_filepath = results_folder + "{0}{1}_{2}.csv".format(
            RESULTS_FILENAME_PREFIX.format(LINEEXTRACTION_TYPE_WATERSHED),
            Path(p_book_directory).name,
            self.config[RUN_UUID]
        )

        with open(stats_filepath, "w") as output_file:

            print(f"Writing stats file for book {Path(stats_filepath).name}")

            csv_writer = csv.writer(output_file)

            csv_writer.writerow([
                "image_name",
                "image_width",
                "image_height",                    
                "image_area",
                "area_all_lines",
                "num_lines",
                "variance_line_height",
                "median_line_height"
            ])

            for image_name in csv_results["images"]:

                csv_writer.writerow([
                    image_name,
                    csv_results["images"][image_name]["image_width"],
                    csv_results["images"][image_name]["image_height"],
                    csv_results["images"][image_name]["image_area"],
                    csv_results["images"][image_name]["area_all_lines"],
                    csv_results["images"][image_name]["num_lines"],
                    csv_results["images"][image_name]["variance_line_height"],
                    csv_results["images"][image_name]["median_line_height"]
                ])

        print("Exiting QA_LineExtraction.__output_stats_on_book_watershed")                
        
        return csv_results

    def __tally_booklevel_stats_watershed(self, p_book_directory, p_pagelevel_stats):

        print("Entering QA_LineExtraction.__tally_booklevel_stats_watershed")

        # 1. Tally page level data for book
        areas_all_lines = []
        image_areas = []
        image_heights = []
        image_widths = []
        line_counts = []
        line_height_medians = []
        line_height_variances = []

        for image_name in p_pagelevel_stats["images"]:

            # Lines

            # Count
            line_counts.append(int(p_pagelevel_stats["images"][image_name]["num_lines"]))

            # Dimensions
            line_height_medians.append(p_pagelevel_stats["images"][image_name]["median_line_height"])
            line_height_variances.append(p_pagelevel_stats["images"][image_name]["variance_line_height"])
            areas_all_lines.append(p_pagelevel_stats["images"][image_name]["area_all_lines"])

            # Images

            # Dimensions
            image_widths.append(p_pagelevel_stats["images"][image_name]["image_width"])
            image_heights.append(p_pagelevel_stats["images"][image_name]["image_height"])
            image_areas.append(p_pagelevel_stats["images"][image_name]["image_area"])

        # 2. Tally errors for book  
        error_filepath = ""
        error_lookup = {}
        results_folder = format_path(p_book_directory + RESULTS_DIRECTORY)
        total_errors = 0
        total_unique_errors = 0

        for filepath in glob.glob(results_folder + WATERSHED_MERGED_ERROR_FILENAME.replace("{}", "*")):
            error_filepath = filepath
            break
        print("ERROR FILEPATH for Book {0}: {1}".format(Path(p_book_directory).name, error_filepath))
        print("results_folder: {0}".format(results_folder))
        print("WATERSHED_MERGED_ERROR_FILENAME search string: {0}".format(WATERSHED_MERGED_ERROR_FILENAME.replace("{}", "*")))

        if error_filepath:
            with open(error_filepath, "r") as merged_error_file:
                csv_reader = csv.DictReader(merged_error_file)
                for row in csv_reader:
                    error_lookup[row["error_source"]] = row["error"]
        
        if len(error_lookup):
            total_errors = sum([len(error_lookup[error_source]) for error_source in error_lookup])
            total_unique_errors = len(list(set([error for error_source in error_lookup for error in error_lookup[error_source]])))

        # 3. Book level stats for results output
        booklevel_stats = {

            "median_area_all_lines": 0,
            "median_image_area": 0,
            "median_image_height": 0,
            "median_image_width": 0,
            "median_line_height_median": 0,
            "median_page_line_count": 0,
            "median_variance_line_height": 0
        }

        if len(p_pagelevel_stats["images"]) > 0:
            booklevel_stats["median_area_all_lines"] = median(areas_all_lines)
            booklevel_stats["median_image_area"] = median(image_areas)
            booklevel_stats["median_image_height"] = median(image_heights)
            booklevel_stats["median_image_width"] = median(image_widths)
            booklevel_stats["median_line_height_median"] = median(line_height_medians)
            booklevel_stats["median_page_line_count"] = median(line_counts)
            booklevel_stats["median_variance_line_height"] = median(line_height_variances)

        booklevel_stats["total_errors"] = total_errors
        booklevel_stats["total_unique_errors"] = total_unique_errors
        booklevel_stats["total_lines"] = sum(line_counts)
        booklevel_stats["total_pages"] = len(p_pagelevel_stats["images"])

        print("Exiting QA_LineExtraction.__tally_booklevel_stats_watershed")

        return booklevel_stats       


    # 'run' command and helpers

    def run(self):
    
        print("Entering QA_LineExtraction.run")

        self.slurm_job_results = []

        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            self.slurm_job_results = self.__run_on_book(self.config[BOOK_DIRECTORY])
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            self.slurm_job_results = self.__run_on_all_books()

        print_debug_header("Slurm Job Results")
        for index in range(len(self.slurm_job_results)):
            print("Result {0}: {1}".format(index, self.slurm_job_results[index]))
        print_debug_header()

        print("Exiting QA_LineExtraction.run")

    def __run_on_all_books(self):

        print("Entering/exiting __run_on_all_books")

        return [ self.__run_on_book(format_path(self.config[BOOK_DIRECTORY] + book_name)) \
            for book_name in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]) \
            if Path(self.config[OUTPUT_DIRECTORY]).name != book_name ]

    def __run_on_book(self, p_book_directory):

        print("Entering __run_on_book")

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

        print("Exiting __run_on_book")           
        
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