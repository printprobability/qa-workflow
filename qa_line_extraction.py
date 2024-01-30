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
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from statistics import median, variance

# Third party
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
DIRECTORY_QA_RESULTS = "qa_results"
LINEEXTRACTION_SCRIPT_LOCATION_DHSEGMENT = QA_CODE_DIRECTORY + "run_dhsegment_on_book.py"
LINEEXTRACTION_SCRIPT_LOCATION_WATERSHED = QA_CODE_DIRECTORY + "watershed_line_extraction.py"

LINEEXTRACTION_TYPE_EYNOLLAH = "eynollah"
LINEEXTRACTION_TYPE_WATERSHED = "watershed"
LINEEXTRACTION_TYPES = [
    LINEEXTRACTION_TYPE_WATERSHED,
    LINEEXTRACTION_TYPE_EYNOLLAH
]

# Watershed files

LINEEXTRACTION_WATERSHED_METADATA_FILE = "line_df.csv"
DHSEGMENT_ERROR_FILENAME = "le_dhsegment_errors_*.txt"
WATERSHED_ERROR_FILENAME = "le_watershed_errors_*.txt"
WATERSHED_MERGED_ERROR_FILENAME_BOOK = "le_watershed_{}_errors_{}.txt"
WATERSHED_MERGED_ERROR_FILENAME_RUN = "le_watershed_all_errors_{}.txt"

# Eynollah files and directories

EYNOLLAH_LINE_IMAGE_EXTRACTION_SCRIPT = "eynollah_line_image_extraction.py"
EYNOLLAH_METADATA_FILE = "line_df.csv"
EYNOLLAH_MODEL_DIRECTORY = "eynollah_model"
EYNOLLAH_OUTPUT_DIRECTORY = "eynollah_output"
EYNOLLAH_EXTRACTED_IMAGES_DIRECTORY = "extracted_images"
EYNOLLAH_PAGE_XML_DIRECTORY = "pagexml"

# QA output

MASTER_LOG_FILENAME_PREFIX = "qa_slurm"
QA_OUTPUT_PREFIX = "le_{}_"
RESULTS_FILENAME_PREFIX = QA_OUTPUT_PREFIX + "results_"
ERRORS_FILENAME_PREFIX = QA_OUTPUT_PREFIX + "errors_"

# sbatch parameters

SBATCH_NTASKS_PER_NODE = "1"
SBATCH_PARTITION = "RM-shared"
SBATCH_TIME = "48:00:00"


# Classes

# NOTE: Core/shared line extraction functionality exists in the QA_LineExtraction
# base class. Behavior more specific to the different line extraction methods
# are separated out into their respective child classes below.

class QA_LineExtraction(QA_Module):

    def __init__(self, p_config):

        print("Entering QA_LineExtraction.__init__")

        self.config = p_config
        self.slurm_job_results = []

        print("Exiting QA_LineExtraction.__init__")

    # 'archive' subcommands

    def archive_logs(self):

        print("Entering QA_LineExtraction.archive_logs")

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
    
    # 'output_stats' command and helpers

    def output_stats(self):

        print("Entering QA_LineExtraction.output_stats")

        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:

            book_name = Path(self.config[BOOK_DIRECTORY]).name
            booklevel_stats = {
                book_name: self._Base__output_stats_on_book(self.config[BOOK_DIRECTORY])
            }
            self.__output_stats_runlevel(booklevel_stats)
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:

            self.__output_stats_on_all_books()

        print("Exiting QA_LineExtraction.output_stats")

    def __output_stats_on_all_books(self):

        print("Entering QA_LineExtraction.__output_stats_on_all_books")
        
        # 1. Output a results file per book and store booklevel stats that are returned
        booklevel_stats = {}
        for book_name in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):
            if DIRECTORY_QA_RESULTS != book_name:
                booklevel_stats[book_name] = self._Base__output_stats_on_book(format_path(self.config[BOOK_DIRECTORY] + book_name))

        # 2. Output one file containing booklevel stats of for whole line extraction run
        self.__output_stats_runlevel(booklevel_stats)

        # 3. Create a master file of all book level stats for this run for all line extraction types
        self.__merge_booklevel_statsfiles()

        print("Exiting QA_LineExtraction.__output_stats_on_all_books")

    @abstractmethod
    def _Base__output_stats_on_book(self, p_book_directory):
        raise NotImplementedError("Must override QA_LineExtraction.__output_stats_on_book")

    # 'run' command and helpers

    def run(self):
    
        print("Entering QA_LineExtraction.run")

        self.slurm_job_results = []

        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            self.slurm_job_results = self._Base__run_on_book(self.config[BOOK_DIRECTORY])
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
    
    @abstractmethod
    def _Base__run_on_book(self, p_book_directory):
        raise NotImplementedError("Must override QA_LineExtraction.__run_on_book")

class QA_LineExtraction_Eynollah(QA_LineExtraction):

    def __init__(self, p_config):

        print("Entering QA_LineExtraction_Eynollah.__init__")

        super().__init__(p_config)

        print("Exiting QA_LineExtraction_Eynollah.__init__")

    # 'clear' subcommands

    def clear_results(self):

        print("Entering QA_LineExtraction_Eynollah.clear_results")

        # 0. All subdirectories in line extraction book directory to be deleted   
        directories_to_be_removed = [

            DIRECTORY_PAGES,
            DIRECTORY_PAGES_COLOR,
            DIRECTORY_LINES,
            DIRECTORY_LINES_COLOR,
            EYNOLLAH_OUTPUT_DIRECTORY,
            DIRECTORY_QA_RESULTS,
        ]

        # 1. Establish list of book directories to look through for subdirectory deletion
        book_directories = []
        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            book_directories.append(self.config[BOOK_DIRECTORY])
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            book_directories.extend([self.config[BOOK_DIRECTORY] + directory + os.sep for directory in get_items_in_dir(format_path(self.config[BOOK_DIRECTORY]), ["directories"])])

        # 2. Delete listed line extraction subdirectories in all book directories
        for book_directory in book_directories:
            for directory in directories_to_be_removed:
                directory_to_be_removed = book_directory + directory
                if os.path.exists(directory_to_be_removed):
                    shutil.rmtree(directory_to_be_removed, ignore_errors=True)
                    wait_while_exists(directory_to_be_removed)

        print("Exiting QA_LineExtraction_Eynollah.clear_results")

    # 'collate' command, subcommands, and helpers

    def collate(self):

        print("Entering QA_LineExtraction_Eynollah.collate")

        # NOTE: This is run *before* output_stats, so that errors can
        # later be outputted into a master file and tallied via the
        # combined error files created by __collate_errors_on_book_watershed
        # QA sequence for watershed is 'clear', 'run', 'collate', 'output_stats'
        
        # self.collate_errors()

        print("Exiting QA_LineExtraction_Eynollah.collate")

    def collate_errors(self, p_le_type):

        print("Entering QA_LineExtraction_Eynollah.collate_errors")

        # 1. Bring together the two error files (eynollah parts 1 and 2) into one error file
        # if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
        #     self.__collate_errors_on_book(self.config[BOOK_DIRECTORY])
        # elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
        #     for book_directory in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):
        #         self.__collate_errors_on_book(format_path(self.config[BOOK_DIRECTORY] + book_directory))
            
        # 2. Merge all errors into one file in the output directory
        # self.__collate_all_errors()

        print("Exiting QA_LineExtraction_Eynollah.collate_errors")

    def __collate_errors_on_book(self, p_book_directory):
    
        print("Entering QA_LineExtraction_Eynollah.__collate_errors_on_book")

        # # 1. Collect error files from both Eynollah parts 1 and 2
        # dh_segment_error_filepath = os.path.join(p_book_directory, DIRECTORY_QA_RESULTS, EYNOLLAH_PART1_ERROR_FILENAME)
        # watershed_error_filepath = os.path.join(p_book_directory, DIRECTORY_QA_RESULTS, EYNOLLAH_PART2_ERROR_FILENAME)
        # error_dict = {

        #     "dh_segment": read_error_file(dh_segment_error_filepath, "LE"),
        #     "watershed": read_error_file(watershed_error_filepath, "LE")
        # }

        # # 2. Create the results directory in the book directory
        # results_directory = p_book_directory + DIRECTORY_QA_RESULTS + os.sep
        # if not os.path.exists(results_directory):
        #     os.makedirs(results_directory)

        # # 3. Output them into one csv file
        # book_name = Path(p_book_directory).name
        # combined_error_filepath = results_directory + \
        #     "{0}{1}_{2}.csv".format(ERRORS_FILENAME_PREFIX.format(LINEEXTRACTION_TYPE_WATERSHED),
        #                             book_name, self.config[RUN_UUID])
        # with open(combined_error_filepath, "w") as merged_error_file:
        #     csv_writer = csv.writer(merged_error_file)
        #     csv_writer.writerow(["error_source", "error"])
        #     for le_submodule in error_dict:
        #         for error_source in error_dict[le_submodule]:
        #             for error_lines in error_dict[le_submodule][error_source]:
        #                 csv_writer.writerow([
        #                     error_source,
        #                     "\n".join(error_lines)
        #                 ])

        print("Exiting QA_LineExtraction_Eynollah.__collate_errors_on_book")

    def __collate_all_errors(self):

        # # 1. List of book directories where error csvs can be found
        # book_directories = []
        # if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
        #     book_directories = [self.config[BOOK_DIRECTORY]]
        # elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
        #     for book_directory in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):
        #         book_directories.append(self.config[BOOK_DIRECTORY] + os.sep + book_directory + os.sep)

        # # 2. Add errors to a master error file in the output directory
        # for book_directory in book_directories:
            
        #     results_directory = book_directory + DIRECTORY_QA_RESULTS + os.sep
        #     combined_error_filepath = results_directory + \
        #         "{0}{1}_{2}.csv".format(ERRORS_FILENAME_PREFIX.format(LINEEXTRACTION_TYPE_EYNOLLAH),
        #                                 Path(book_directory).name, self.config[RUN_UUID])
        #     merge_all_filepath = self.config[OUTPUT_DIRECTORY] + EYNOLLAH_MERGED_ERROR_FILENAME_RUN.format(self.config[RUN_UUID])
            
        #     # A. Read in errors from this file
        #     error_lookup = {}
        #     with open(combined_error_filepath, "r") as input_file:
        #         csv_reader = csv.DictReader(input_file)
        #         for row in csv_reader:
        #             error_lookup[row["error_source"]] = row["error"]

        #     # B. Add the errors to the merged master error file
        #     write_header = False
        #     if not os.path.exists(merge_all_filepath):
        #         write_header = True

        #     with open(merge_all_filepath, "a") as output_file:

        #         if write_header:
        #             output_file.write("error_source,error\n")

        #         for error_source in error_lookup:
        #             output_file.write(f"{error_source},{error_lookup[error_source]}\n")

        pass

    # 'output_stats' helpers

    def _Base__output_stats_on_book(self, p_book_directory):

        print("Entering QA_LineExtraction_Eynollah.__output_stats_on_book")

        csv_results = {}

        # 1. Gather and output stats for lines for the eynollah line extraction method
        csv_results["page"] = self.__output_stats_on_book_eynollah(p_book_directory)

        # 2. Tally book stats for potential use outside of function
        csv_results["book"] = self.__tally_booklevel_stats(csv_results["page"])

        print("Exiting QA_LineExtraction_Eynollah.__output_stats_on_book")

        # 3. Return just book level stats for each line extraction type
        return csv_results

    def __output_stats_on_book_eynollah(self, p_book_directory):

        print("Entering QA_LineExtraction_Eynollah.__output_stats_on_book_eynollah")        

        csv_results = { "images": {} }
        lines_color_folder = format_path(p_book_directory + DIRECTORY_LINES_COLOR)
        line_df_filepath = lines_color_folder + EYNOLLAH_METADATA_FILE
        pages_color_folder = p_book_directory + DIRECTORY_PAGES_COLOR
        results_folder = format_path(p_book_directory + DIRECTORY_QA_RESULTS)

        # 0. Make a folder for the new output stats file
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # 1. Potential error file for this line extraction run for this book
        # TODO: Writing error processing hooks in eynollah line extraction scripts to output error file
            
        # 2. Determine info about the lines extracted for the book pages
        if not os.path.exists(line_df_filepath):

            print(f"ERROR: Could not find {EYNOLLAH_METADATA_FILE} for {Path(p_book_directory).name}")
            print("Exiting QA_LineExtraction.__output_stats_on_book_watershed")
            return csv_results
        
        print(f"line_df_filepath: {line_df_filepath}")
        with open(line_df_filepath, "r") as line_df_file:

            csv_reader = csv.DictReader(line_df_file)

            # A. Store information for each line on each page
            for line_row in csv_reader:

                print(f"Examining {EYNOLLAH_METADATA_FILE} row: {line_row}")

                file_name_parts = line_row["file_name"].split("_page1r_")
                image_name = file_name_parts[0]
                line_number = file_name_parts[1]

                if image_name not in csv_results["images"]:
                    csv_results["images"][image_name] = { "lines": {} }

                # I. Save angle of rotation, width, and height
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

                # a. Determine if dimensions need to be flipped due to angle
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
                
                # II. Save norm height
                csv_results["images"][image_name]["lines"][line_number]["norm_height"] = line_row["norm_height"]

                # III. Save any errors from error lookup for this line
                # NOTE: Uncomment when error processing is enabled
                # csv_results["images"][image_name]["lines"][line_number]["error(s)"] = "N/A"
                # if line_row["file_name"] in error_lookup:
                #     csv_results["images"][image_name]["lines"][line_number]["error(s)"] = error_lookup[line_row["file_name"]]

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

                # VI. Median norm height
                norm_heights = [
                    float(csv_results["images"][image_name]["lines"][line_number]["norm_height"])
                    for line_number in csv_results["images"][image_name]["lines"]      
                ]
                csv_results["images"][image_name]["median_norm_height"] = median(norm_heights)

        # 3. Output a csv file of these stats in the line extraction results folder    
        stats_filepath = results_folder + "{0}{1}_{2}.csv".format(
            RESULTS_FILENAME_PREFIX.format(LINEEXTRACTION_TYPE_EYNOLLAH),
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
                "median_line_height",
                "median_norm_height"
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
                    csv_results["images"][image_name]["median_line_height"],
                    csv_results["images"][image_name]["median_norm_height"]
                ])

        print("Exiting QA_LineExtraction_Eynollah.__output_stats_on_book_eynollah")                
        
        return csv_results
    
    def __tally_booklevel_stats(self, p_pagelevel_stats):

        print("Entering QA_LineExtraction_Eynollah.__tally_booklevel_stats_eynollah")

        # 1. Tally page level data for book
        areas_all_lines = []
        image_areas = []
        image_heights = []
        image_widths = []
        line_counts = []
        line_height_medians = []
        line_height_variances = []
        line_norm_height_medians = []

        for image_name in p_pagelevel_stats["images"]:

            # Lines

            # Count
            line_counts.append(int(p_pagelevel_stats["images"][image_name]["num_lines"]))

            # Dimensions
            line_height_medians.append(p_pagelevel_stats["images"][image_name]["median_line_height"])
            line_height_variances.append(p_pagelevel_stats["images"][image_name]["variance_line_height"])
            areas_all_lines.append(p_pagelevel_stats["images"][image_name]["area_all_lines"])
            line_norm_height_medians.append(p_pagelevel_stats["images"][image_name]["median_norm_height"])

            # Images

            # Dimensions
            image_widths.append(p_pagelevel_stats["images"][image_name]["image_width"])
            image_heights.append(p_pagelevel_stats["images"][image_name]["image_height"])
            image_areas.append(p_pagelevel_stats["images"][image_name]["image_area"])

        # # 2. Tally errors for book
        # NOTE: Uncomment this when error processing enabled for eynollah line extraction
        # error_filepath = ""
        # error_lookup = {}
        # results_folder = format_path(p_book_directory + DIRECTORY_QA_RESULTS)
        # total_errors = 0
        # total_unique_errors = 0

        # for filepath in glob.glob(results_folder + WATERSHED_MERGED_ERROR_FILENAME_BOOK.replace("{}", "*")):
        #     error_filepath = filepath
        #     break
        # print("ERROR FILEPATH for Book {0}: {1}".format(Path(p_book_directory).name, error_filepath))
        # print("results_folder: {0}".format(results_folder))
        # print("WATERSHED_MERGED_ERROR_FILENAME_BOOK search string: {0}".format(WATERSHED_MERGED_ERROR_FILENAME_BOOK.replace("{}", "*")))
        # print("self.config: {0}".format(self.config))
        # error_filepath = results_folder + WATERSHED_MERGED_ERROR_FILENAME_BOOK.format(
        #     Path(p_book_directory).name,
        #     self.config[ERROR_FILE_RUN_UUID]
        # )
        # print("OVERRIDING error filepath: {0}".format(error_filepath))
        # print("File exists: {0}".format(os.path.exists(error_filepath)))
        # if not os.path.exists(error_filepath):
        #     error_filepath = ""
        # else:
        #     print("FOUND FOUND FOUND FOUND FOUND FOUND FOUND")
        #     print("error_lookup at: {0}".format(error_filepath))

        # if error_filepath:
        #     with open(error_filepath, "r") as merged_error_file:
        #         csv_reader = csv.DictReader(merged_error_file)
        #         for row in csv_reader:
        #             if row["error_source"] not in error_lookup:
        #                 error_lookup[row["error_source"]] = []
        #             error_lookup[row["error_source"]].append(row["error"])

        # print("Error lookup key count: {0}".format(len(list(error_lookup.keys()))))
        
        # if len(list(error_lookup.keys())):

        #     total_errors = sum([len(error_lookup[error_source]) for error_source in error_lookup])
        #     total_unique_errors = len(list(set([error for error_source in error_lookup for error in error_lookup[error_source]])))

        #     print("Total errors: {0}".format(total_errors))
        #     print("Total unique errors: {0}".format(total_unique_errors))

        # 3. Book level stats for results output
        booklevel_stats = {

            "median_area_all_lines": 0,
            "median_image_area": 0,
            "median_image_height": 0,
            "median_image_width": 0,
            "median_line_height_median": 0,
            "median_line_norm_height_median": 0,
            "median_page_line_count": 0,
            "median_variance_line_height": 0
        }

        if len(p_pagelevel_stats["images"]) > 0:
            booklevel_stats["median_area_all_lines"] = median(areas_all_lines)
            booklevel_stats["median_image_area"] = median(image_areas)
            booklevel_stats["median_image_height"] = median(image_heights)
            booklevel_stats["median_image_width"] = median(image_widths)
            booklevel_stats["median_line_height_median"] = median(line_height_medians)
            booklevel_stats["median_line_norm_height_median"] = median(line_norm_height_medians)
            booklevel_stats["median_page_line_count"] = median(line_counts)
            booklevel_stats["median_variance_line_height"] = median(line_height_variances)

        # NOTE: Uncomment when error processing for eynollah is enabled
        # booklevel_stats["total_errors"] = total_errors
        # booklevel_stats["total_unique_errors"] = total_unique_errors

        booklevel_stats["total_lines"] = sum(line_counts)
        booklevel_stats["total_pages"] = len(p_pagelevel_stats["images"])

        print("Exiting QA_LineExtraction_Eynollah.__tally_booklevel_stats_eynollah")

        return booklevel_stats

    def __merge_booklevel_statsfiles(self):

        print("Entering QA_LineExtraction_Eynollah.__merge_booklevel_statsfiles")

        master_stats_filename = "{0}{1}.csv".format(RESULTS_FILENAME_PREFIX.format(LINEEXTRACTION_TYPE_EYNOLLAH), self.config[RUN_UUID])

        with open(self.config[OUTPUT_DIRECTORY] + master_stats_filename, "w") as output_file:

            # 1. Read the outputted stats file for each book and write it to the master stats file
            header_written = False
            for book_directory in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):

                results_directory = format_path(self.config[BOOK_DIRECTORY] + book_directory + os.sep + DIRECTORY_QA_RESULTS)
                bookstats_filename =  "{0}{1}_{2}.csv".format(
                    RESULTS_FILENAME_PREFIX.format(LINEEXTRACTION_TYPE_EYNOLLAH),
                    book_directory,
                    self.config[RUN_UUID]
                )

                if not os.path.exists(results_directory + bookstats_filename):
                    print("ERROR: Stats file does not exist for {0} at {1}".format(book_directory, results_directory + bookstats_filename))
                    continue 

                # Add the book stats file contents to the master stats file (skipping the header if already written)
                with open(results_directory + bookstats_filename, "r") as input_file:
                    
                    output_file.writelines(input_file.readlines()[1:] if header_written else input_file.readlines())
                    header_written = True

        print("Exiting QA_LineExtraction_Eynollah.__merge_booklevel_statsfiles_watershed")

    def _QA_LineExtraction__output_stats_runlevel(self, p_booklevel_stats):

        print("Entering QA_LineExtraction_Eynollah.__output_stats_runlevel")

        results_filename_prefix = RESULTS_FILENAME_PREFIX.format(LINEEXTRACTION_TYPE_EYNOLLAH)
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
                "median_variance_line_height",
                "median_line_norm_height_median"
            ])

            for book_name in p_booklevel_stats:

                print(f"Book level stats for {book_name}:")
                print(f"{p_booklevel_stats[book_name]}")                

                csv_writer.writerow([

                    book_name,
                    p_booklevel_stats[book_name]["book"]["total_pages"],
                    p_booklevel_stats[book_name]["book"]["total_lines"],
                    "N/A",# p_booklevel_stats[book_name]["total_errors"],
                    "N/A", # p_booklevel_stats[book_name]["total_unique_errors"],
                    p_booklevel_stats[book_name]["book"]["median_image_width"],
                    p_booklevel_stats[book_name]["book"]["median_image_height"],
                    p_booklevel_stats[book_name]["book"]["median_image_area"],
                    p_booklevel_stats[book_name]["book"]["median_area_all_lines"],
                    p_booklevel_stats[book_name]["book"]["median_line_height_median"],
                    p_booklevel_stats[book_name]["book"]["median_variance_line_height"],
                    p_booklevel_stats[book_name]["book"]["median_line_norm_height_median"]
                ])                    

        print("Exiting QA_LineExtraction_Eynollah.__output_stats_runlevel")

    # 'run' helpers

    def _Base__run_on_book(self, p_book_directory):

        print("Entering QA_LineExtraction_Eynollah.__run_on_book")

        book_name = Path(p_book_directory).name
        slurm_results = []       
        
        # 1. Start up slurm jobs to test the Eynollah line extraction methods on this book        
        print("Creating slurm job for QA of line extraction {0} with line extraction type {1}".format(book_name, LINEEXTRACTION_TYPE_EYNOLLAH))

        # A. Run line extraction on the book with the given arguments

        # I. sbatch arguments
        sbatch_directives = {

            "-ntasks-per-node": SBATCH_NTASKS_PER_NODE,
            "-o": "{0}slurm-{1}_{2}_{3}.out".format(
                self.config[OUTPUT_DIRECTORY],
                book_name,
                LINEEXTRACTION_TYPE_EYNOLLAH,
                self.config[RUN_UUID]),
            "-p": SBATCH_PARTITION,                
            "-t": SBATCH_TIME
        }        

        # II. Build the sbatch call
        subprocess_cmd = "sbatch"
        for arg in sbatch_directives:
            subprocess_cmd += " {0} {1}".format(arg, sbatch_directives[arg])
        subprocess_cmd += " {0}{1}qa_line_extraction_eynollah.sh {2} {3} {4}".format(
            os.getcwd(), os.sep,
            p_book_directory, self.config[RUN_UUID])

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

        print("Exiting QA_LineExtraction_Eynollah.__run_on_book")           
        
        return slurm_results

class QA_LineExtraction_Watershed(QA_LineExtraction):

    # 'clear' subcommands

    def clear_results(self):

        print("Entering QA_LineExtraction_Watershed.clear_results")

        # 0. All subdirectories in line extraction book directory to be deleted
        directories_to_be_removed = [

            DIRECTORY_PAGES,
            DIRECTORY_PAGES_COLOR,
            DIRECTORY_LINES,
            DIRECTORY_LINES_COLOR,
            DIRECTORY_QA_RESULTS
        ]

        # 1. Establish list of book directories to look through for subdirectory deletion
        book_directories = []
        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            book_directories.append(self.config[BOOK_DIRECTORY])
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            book_directories.extend([self.config[BOOK_DIRECTORY] + directory + os.sep for directory in get_items_in_dir(format_path(self.config[BOOK_DIRECTORY]), ["directories"])])

        # 2. Delete listed line extraction subdirectories in all book directories
        for book_directory in book_directories:
            for directory in directories_to_be_removed:
                directory_to_be_removed = book_directory + directory
                if os.path.exists(directory_to_be_removed):
                    shutil.rmtree(directory_to_be_removed, ignore_errors=True)
                    wait_while_exists(directory_to_be_removed)

        print("Exiting QA_LineExtraction_Watershed.clear_results")

    # 'collate' command, subcommands, and helpers

    def collate(self):

        print("Entering QA_LineExtraction_Watershed.collate")

        # NOTE: This is run *before* output_stats, so that errors can
        # later be outputted into a master file and tallied via the
        # combined error files created by __collate_errors_on_book_watershed
        # QA sequence for watershed is 'clear', 'run', 'collate', 'output_stats'
        self.collate_errors(LINEEXTRACTION_TYPE_WATERSHED)

        print("Exiting QA_LineExtraction_Watershed.collate")

    def collate_errors(self, p_le_type):

        print("Entering QA_LineExtraction_Watershed.collate_errors")

        # 1. Bring together the two error files (dhsegment and watershed) into one error file
        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            self.__collate_errors_on_book(self.config[BOOK_DIRECTORY])
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            for book_directory in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):
                self.__collate_errors_on_book(format_path(self.config[BOOK_DIRECTORY] + book_directory))
            
        # 2. Merge all errors into one file in the output directory
        self.__collate_all_errors()

        print("Exiting QA_LineExtraction_Watershed.collate_errors")

    def __collate_errors_on_book(self, p_book_directory):
    
        print("Entering QA_LineExtraction_Watershed.__collate_errors_on_book")

        # 1. Collect error files from both the run_dhsegment run and the watershed run
        dh_segment_error_filepath = os.path.join(p_book_directory, DIRECTORY_QA_RESULTS, DHSEGMENT_ERROR_FILENAME)
        watershed_error_filepath = os.path.join(p_book_directory, DIRECTORY_QA_RESULTS, WATERSHED_ERROR_FILENAME)
        error_dict = {

            "dh_segment": read_error_file(dh_segment_error_filepath, "LE"),
            "watershed": read_error_file(watershed_error_filepath, "LE")
        }

        # 2. Create the results directory in the book directory
        results_directory = p_book_directory + DIRECTORY_QA_RESULTS + os.sep
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        # 3. Output them into one csv file
        book_name = Path(p_book_directory).name
        combined_error_filepath = results_directory + \
            "{0}{1}_{2}.csv".format(ERRORS_FILENAME_PREFIX.format(LINEEXTRACTION_TYPE_WATERSHED),
                                    book_name, self.config[RUN_UUID])
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

        print("Exiting QA_LineExtraction_Watershed.__collate_errors_on_book")

    def __collate_all_errors(self):

        print("Entering QA_LineExtraction_Watershed.__collate_all_errors")

        # 1. List of book directories where error csvs can be found
        book_directories = []
        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            book_directories = [self.config[BOOK_DIRECTORY]]
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            for book_directory in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):
                book_directories.append(self.config[BOOK_DIRECTORY] + os.sep + book_directory + os.sep)

        # 2. Add errors to a master error file in the output directory
        for book_directory in book_directories:
            
            results_directory = book_directory + DIRECTORY_QA_RESULTS + os.sep
            combined_error_filepath = results_directory + \
                "{0}{1}_{2}.csv".format(ERRORS_FILENAME_PREFIX.format(LINEEXTRACTION_TYPE_WATERSHED),
                                        Path(book_directory).name, self.config[RUN_UUID])
            merge_all_filepath = self.config[OUTPUT_DIRECTORY] + WATERSHED_MERGED_ERROR_FILENAME_RUN.format(self.config[RUN_UUID])
            
            # A. Read in errors from this file
            error_lookup = {}
            with open(combined_error_filepath, "r") as input_file:
                csv_reader = csv.DictReader(input_file)
                for row in csv_reader:
                    error_lookup[row["error_source"]] = row["error"]

            # B. Add the errors to the merged master error file
            write_header = False
            if not os.path.exists(merge_all_filepath):
                write_header = True

            with open(merge_all_filepath, "a") as output_file:

                if write_header:
                    output_file.write("error_source,error\n")

                for error_source in error_lookup:
                    output_file.write(f"{error_source},{error_lookup[error_source]}\n")

        print("Exiting QA_LineExtraction_Watershed.__collate_all_errors")
                
    # 'output_stats' helpers

    def _Base__output_stats_on_book(self, p_book_directory):

        print("Entering QA_LineExtraction_Watershed.__output_stats_on_book")

        csv_results = {}

        # 1. Gather and output stats for lines for the watershed line extraction method
        csv_results["page"] = self.__output_stats_on_book_watershed(p_book_directory)

        # 2. Tally book stats for potential use outside of function
        csv_results["book"] = self.__tally_booklevel_stats_watershed(p_book_directory, csv_results[le_type]["page"])

        print("Exiting QA_LineExtraction_Watershed.__output_stats_on_book")

        # 3. Return just book level stats for each line extraction type
        return csv_results

    def __output_stats_on_book_watershed(self, p_book_directory):

        print("Entering QA_LineExtraction_Watershed.__output_stats_on_book_watershed")

        csv_results = { "images": {} }
        lines_color_folder = format_path(p_book_directory + DIRECTORY_LINES_COLOR)
        line_df_filepath = lines_color_folder + LINEEXTRACTION_WATERSHED_METADATA_FILE
        pages_color_folder = p_book_directory + DIRECTORY_PAGES_COLOR
        results_folder = format_path(p_book_directory + DIRECTORY_QA_RESULTS)

        # 0. Make a folder for the new output stats file
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # 1. Potential error file for this line extraction run for this book
        error_lookup = {}
        error_filepath = ""
        for filepath in glob.glob(results_folder + WATERSHED_MERGED_ERROR_FILENAME_BOOK.replace("{}", "*")):
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

            # A. Store information for each line on each page
            for line_row in csv_reader:

                file_name_parts = line_row["file_name"].split("_page1r_")
                image_name = file_name_parts[0]
                line_number = file_name_parts[1]

                if image_name not in csv_results["images"]:
                    csv_results["images"][image_name] = { "lines": {} }

                # I. Save angle of rotation, width, and height
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

                # a. Determine if dimensions need to be flipped due to angle
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

                # II. Save any errors from error lookup for this line
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

        print("Exiting QA_LineExtraction_Watershed.__output_stats_on_book_watershed")                
        
        return csv_results

    def __tally_booklevel_stats_watershed(self, p_book_directory, p_pagelevel_stats):

        print("Entering QA_LineExtraction_Watershed.__tally_booklevel_stats_watershed")

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
        results_folder = format_path(p_book_directory + DIRECTORY_QA_RESULTS)
        total_errors = 0
        total_unique_errors = 0

        for filepath in glob.glob(results_folder + WATERSHED_MERGED_ERROR_FILENAME_BOOK.replace("{}", "*")):
            error_filepath = filepath
            break
        print("ERROR FILEPATH for Book {0}: {1}".format(Path(p_book_directory).name, error_filepath))
        print("results_folder: {0}".format(results_folder))
        print("WATERSHED_MERGED_ERROR_FILENAME_BOOK search string: {0}".format(WATERSHED_MERGED_ERROR_FILENAME_BOOK.replace("{}", "*")))
        print("self.config: {0}".format(self.config))
        error_filepath = results_folder + WATERSHED_MERGED_ERROR_FILENAME_BOOK.format(
            Path(p_book_directory).name,
            self.config[ERROR_FILE_RUN_UUID]
        )
        print("OVERRIDING error filepath: {0}".format(error_filepath))
        print("File exists: {0}".format(os.path.exists(error_filepath)))
        if not os.path.exists(error_filepath):
            error_filepath = ""
        else:
            print("FOUND FOUND FOUND FOUND FOUND FOUND FOUND")
            print("error_lookup at: {0}".format(error_filepath))

        if error_filepath:
            with open(error_filepath, "r") as merged_error_file:
                csv_reader = csv.DictReader(merged_error_file)
                for row in csv_reader:
                    if row["error_source"] not in error_lookup:
                        error_lookup[row["error_source"]] = []
                    error_lookup[row["error_source"]].append(row["error"])

        print("Error lookup key count: {0}".format(len(list(error_lookup.keys()))))
        
        if len(list(error_lookup.keys())):

            total_errors = sum([len(error_lookup[error_source]) for error_source in error_lookup])
            total_unique_errors = len(list(set([error for error_source in error_lookup for error in error_lookup[error_source]])))

            print("Total errors: {0}".format(total_errors))
            print("Total unique errors: {0}".format(total_unique_errors))

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

        print("Exiting QA_LineExtraction_Watershed.__tally_booklevel_stats_watershed")

        return booklevel_stats      

    def __merge_booklevel_statsfiles(self):

        print("Entering QA_LineExtraction_Watershed.__merge_booklevel_statsfiles")

        master_stats_filename = "{0}{1}.csv".format(RESULTS_FILENAME_PREFIX.format(LINEEXTRACTION_TYPE_WATERSHED), self.config[RUN_UUID])

        with open(self.config[OUTPUT_DIRECTORY] + master_stats_filename, "w") as output_file:

            # 1. Read the outputted stats file for each book and write it to the master stats file
            header_written = False
            for book_directory in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):

                results_directory = format_path(self.config[BOOK_DIRECTORY] + book_directory + os.sep + DIRECTORY_QA_RESULTS)
                bookstats_filename =  "{0}{1}_{2}.csv".format(
                    RESULTS_FILENAME_PREFIX.format(LINEEXTRACTION_TYPE_WATERSHED),
                    book_directory,
                    self.config[RUN_UUID]
                )

                if not os.path.exists(results_directory + bookstats_filename):
                    print("ERROR: Stats file does not exist for {0} at {1}".format(book_directory, results_directory + bookstats_filename))
                    continue 

                # Add the book stats file contents to the master stats file (skipping the header if already written)
                with open(results_directory + bookstats_filename, "r") as input_file:
                    
                    output_file.writelines(input_file.readlines()[1:] if header_written else input_file.readlines())
                    header_written = True

        print("Exiting QA_LineExtraction_Watershed.__merge_booklevel_statsfiles")

    def _QA_LineExtraction__output_stats_runlevel(self, p_booklevel_stats):

        print("Entering QA_LineExtraction_Watershed.__output_stats_runlevel")

        results_filename_prefix = RESULTS_FILENAME_PREFIX.format(LINEEXTRACTION_TYPE_WATERSHED)
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
                    p_booklevel_stats[book_name]["total_pages"],
                    p_booklevel_stats[book_name]["total_lines"],
                    p_booklevel_stats[book_name]["total_errors"],
                    p_booklevel_stats[book_name]["total_unique_errors"],
                    p_booklevel_stats[book_name]["median_image_width"],
                    p_booklevel_stats[book_name]["median_image_height"],
                    p_booklevel_stats[book_name]["median_image_area"],
                    p_booklevel_stats[book_name]["median_area_all_lines"],
                    p_booklevel_stats[book_name]["median_line_height_median"],
                    p_booklevel_stats[book_name]["median_variance_line_height"]
                ])

        print("Exiting QA_LineExtraction_Watershed.__output_stats_runlevel")

    # 'run' helpers

    def _Base__run_on_book(self, p_book_directory):

        print("Entering QA_LineExtraction_Watershed.__run_on_book")

        book_name = Path(p_book_directory).name
        slurm_results = []       
        
        # 1. Start up slurm jobs to test watershed line extraction on this book
        
        print("Creating slurm job for QA of line extraction {0} with line extraction type {1}".format(book_name, LINEEXTRACTION_TYPE_WATERSHED))

        # A. Run line extraction on the book with the given arguments

        # I. sbatch arguments
        sbatch_directives = {

            "--ntasks-per-node": SBATCH_NTASKS_PER_NODE,
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

        print("Exiting QA_LineExtraction_Watershed.__run_on_book")           
        
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

# def run_line_extraction_eynollah_unused(p_args):

#     if RUN_TYPE_SINGLE == qa_config[RUN_TYPE]:

#     subprocess_args = [

#         "eynollah",
#         "-m",
#         QA_CODE_DIRECTORY + os.sep + EYNOLLAH_MODEL_DIRECTORY,
#         "-di",
#         DIRECTORY_PAGES_COLOR,
#         "-o",
#         EYNOLLAH_OUTPUT_DIRECTORY,
#         "-ep",
#         "-cl",
#         "-sa",
#         EYNOLLAH_OUTPUT_DIRECTORY,
#         "-si",
#         EYNOLLAH_OUTPUT_DIRECTORY + os.sep + EYNOLLAH_EXTRACTED_IMAGES_DIRECTORY
#     ] 

# def run_line_extraction_eynollah_old(args):

#     # 0. Show start time
#     print(f"Eynollah line extraction start time: {datetime.now()}")

#     book_directory = format_path(args.book_directory)
#     book_name = Path(book_directory).name

#     # 1. Prepare directory for line extraction and its QA
#     print(f"Preparing directory {book_name} for {LINEEXTRACTION_TYPE_EYNOLLAH} line extraction QA...")

#     # A. Make subdirectories that will be used by the line extraction script
#     print("Making necessary subdirectories...")
#     makedirs(book_directory + DIRECTORY_PAGES)
#     makedirs(book_directory + DIRECTORY_PAGES_COLOR)
#     makedirs(book_directory + DIRECTORY_LINES)
#     makedirs(book_directory + DIRECTORY_LINES_COLOR)
#     makedirs(book_directory + EYNOLLAH_OUTPUT_DIRECTORY)
#     makedirs(book_directory + EYNOLLAH_OUTPUT_DIRECTORY + os.sep + EYNOLLAH_EXTRACTED_IMAGES_DIRECTORY)
#     makedirs(book_directory + EYNOLLAH_OUTPUT_DIRECTORY + os.sep + EYNOLLAH_PAGE_XML_DIRECTORY)

#     # B. Copy the page images to the line extraction subdirectories
#     for image_filepath in glob.glob(book_directory + "*.tif"):
#         shutil.copy(image_filepath, book_directory + DIRECTORY_PAGES)
#         shutil.copy(image_filepath, book_directory + DIRECTORY_PAGES_COLOR)
    
#     # 2. Run Eynollah line extraction on all color pages
#     print(f"Running eynollah line extraction on {book_directory}...")
    
#     # A. Build subprocess command
#     subprocess_args = [

#         "eynollah",
#         "-m",
#         QA_CODE_DIRECTORY + os.sep + EYNOLLAH_MODEL_DIRECTORY,
#         "-di",
#         DIRECTORY_PAGES_COLOR,
#         "-o",
#         EYNOLLAH_OUTPUT_DIRECTORY,
#         "-ep",
#         "-cl",
#         "-sa",
#         EYNOLLAH_OUTPUT_DIRECTORY,
#         "-si",
#         EYNOLLAH_OUTPUT_DIRECTORY + os.sep + EYNOLLAH_EXTRACTED_IMAGES_DIRECTORY
#     ]   

#     # B. Run Eynollah line extraction
#     print(" ".join(subprocess_args))
#     subprocess.run(subprocess_args)

#     # C. Move output xml files to 'pagexml' directory
#     for xml_filepath in glob.glob(book_directory + os.sep + EYNOLLAH_OUTPUT_DIRECTORY + "*.xml"):
#         shutil.move(xml_filepath, EYNOLLAH_OUTPUT_DIRECTORY + os.sep + EYNOLLAH_PAGE_XML_DIRECTORY)
    
#     print(f"Done running eynollah line extraction on {book_directory}.")

#     # NOTE: Temporarily leave out this step for initial testing of eynollah method
#     line_image_extraction = False

#     if line_image_extraction:

#         # 3. Extract minAreaRect lines from files in 'pagexml'
#         # NOTE: Extracts both b/w and color lines from pages/pages_color subdirectories.
#         # We filter out lines above a certain height and below a certain width,
#         # and we standardize line heights for Ocular.

#         print("Starting line extraction on eynollah xml output...")
        
#         # A. Build subprocess command
#         subprocess_args = [

#             "python3",
#             "-u",
#             os.path.join(QA_CODE_DIRECTORY, EYNOLLAH_LINE_IMAGE_EXTRACTION_SCRIPT),
#             EYNOLLAH_OUTPUT_DIRECTORY,
#             "line_minarearect_coords.csv",
#             DIRECTORY_PAGES_COLOR,
#             DIRECTORY_PAGES,
#             "--lines_output_directory",
#             DIRECTORY_LINES,
#             "--color_lines_output_directory",
#             DIRECTORY_LINES_COLOR,
#             "--ext",
#             "tif"
#         ]

#         # B. Run Eynollah line image extraction script
#         print(" ".join(subprocess_args))
#         subprocess.run(subprocess_args)

#     # 4. Show end time
#     print(f"Eynollah line extraction end time: {datetime.now()}")

def run_line_extraction_watershed(args):

    book_directory = format_path(args.book_directory)
    book_name = Path(book_directory).name

    # 1. Prepare directory for line extraction and its QA
    print(f"Preparing directory {book_name} for {LINEEXTRACTION_TYPE_WATERSHED} line extraction QA...")
    
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
    elif LINEEXTRACTION_TYPE_EYNOLLAH == args.line_extraction_type:
        run_line_extraction_eynollah(args)