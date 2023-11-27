# Author: Jonathan Armoza
# Created: October 11, 2023
# Purpose: Tests the autocropper on the test set of books created by create_autocrop_test_dir.py.

# Imports

# Built-ins
import csv
import glob
import os
import shutil
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
MASTER_LOG_FILENAME_PREFIX = "qa_slurm"
MERGED_RESULTS_FILENAME_PREFIX = "autocrop_all_results_merged"
STATS_FILE_PREFIX = "autocrop_results"


# sbatch parameters
SBATCH_MEMORY_PER_CPU = "1999mb"
SBATCH_NUMBER_CPUS = "2"
SBATCH_PARTITION = "RM-shared"
SBATCH_TIME = "06:00:00"


# Classes

class QA_Autocrop(QA_Module):

    def __init__(self, p_config):

        print("Entering QA_Autocrop.__init__")

        super().__init__(p_config)
        self.slurm_job_results = []

        print("Exiting QA_Autocrop.__init__")

    def archive_logs(self):

        print("Entering QA_Autocrop.archive_logs")

        # 0. Make an 'archive' folder in the output directory if it does not exist
        if not os.path.exists(self.config[OUTPUT_DIRECTORY] + ARCHIVE_DIRECTORY):
            os.makedirs(self.config[OUTPUT_DIRECTORY] + ARCHIVE_DIRECTORY)

        # 1. Move each item (that's not the archive folder or master log for this run) into the archive folder
        for item in get_items_in_dir(self.config[OUTPUT_DIRECTORY], ["directories", "files"]):
            if ARCHIVE_DIRECTORY != item and MASTER_LOG_FILENAME_PREFIX not in item and ".gitignore" != item:
                os.rename(self.config[OUTPUT_DIRECTORY] + item,
                    "{0}archive{1}{2}".format(self.config[OUTPUT_DIRECTORY], os.sep, item))
        
        print("Exiting QA_Autocrop.archive_logs")

    def clear_logs(self):

        print("Entering QA_Autocrop.clear_logs")

        # Clear all log files in the log folder
        log_filepaths = glob.glob(self.config[OUTPUT_DIRECTORY] + "*.out")
        for filepath in log_filepaths:
            if MASTER_LOG_FILENAME_PREFIX not in filepath and \
               ".gitignore" not in filepath:
                os.unlink(filepath)
                wait_while_exists(filepath)

        print("Exiting QA_Autocrop.clear_logs")

    def clear_results(self):

        print("Entering QA_Autocrop.clear_results")

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

        print("Exiting QA_Autocrop.clear_results")

    def collate(self):

        print("Entering QA_Autocrop.collate")
        
        # 0. Wait for run and output_stats to finish
        # self.wait()

        # 1. Collate errors and results into two files
        super().collate()

        print("Exiting QA_Autocrop.collate")

    def collate_errors(self):

        print("Entering QA_Autocrop.collate_errors")

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

        print("Exiting QA_Autocrop.collate_errors")

    def collate_results(self):

        print("Entering QA_Autocrop.collate_results")

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

        print("Exiting QA_Autocrop.collate_results")

    def __collate_all_book_results(self):

        print("Entering QA_Autocrop.__collate_all_book_results")

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
        
        print("Exiting QA_Autocrop.__collate_all_book_results")

    def __collate_results_on_book(self, p_results_directory):

        print("Entering QA_Autocrop.__collate_results_on_book")

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
        
        print("Exiting QA_Autocrop.__collate_results_on_book")

    def is_method_finished(self, p_book_directory):

        print("Entering QA_Autocrop.is_method_finished")

        results_folder = format_path(p_book_directory + RESULTS_DIRECTORY)

        # for autocrop_type in AUTOCROP_TYPES:
        #     results_filename = "{0}_{1}_{2}.csv".format(STATS_FILE_PREFIX, autocrop_type, self.config[RUN_UUID])
        #     error_filename = "error_{0}_{1}_{2}.txt".format(Path(p_book_directory).name, autocrop_type, self.config[RUN_UUID])
        #     print("Results file {0}".format(results_folder + results_filename))
        #     print("exists: {0}".format(os.path.exists(results_folder + results_filename)))
        #     print("Error file {0}".format(results_folder + error_filename))
        #     print("exists: {0}".format(os.path.exists(results_folder + error_filename)))

        # print("Method passes: {0}".format( all([
        #     os.path.exists(results_folder + "{0}_{1}_{2}.csv".format(STATS_FILE_PREFIX, autocrop_type, self.config[RUN_UUID])) or \
        #     os.path.exists(results_folder + "error_{0}_{1}_{2}.txt".format(Path(p_book_directory).name, autocrop_type, self.config[RUN_UUID])) \
        #     for autocrop_type in AUTOCROP_TYPES])))

        print("Exiting QA_Autocrop.is_method_finished")

        # Look for a finished results csv for this cropping run or an error file for it
        return all([
            os.path.exists(results_folder + "{0}_{1}_{2}.csv".format(STATS_FILE_PREFIX, autocrop_type, self.config[RUN_UUID])) or \
            os.path.exists(results_folder + "error_{0}_{1}_{2}.txt".format(Path(p_book_directory).name, autocrop_type, self.config[RUN_UUID])) \
            for autocrop_type in AUTOCROP_TYPES])

    def run(self):

        print("Entering QA_Autocrop.run")

        # 1. Run autocrop on book or all books
        self.slurm_job_results = []
        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            self.slurm_job_results = self.__run_autocrop_on_book(self.config[BOOK_DIRECTORY])
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            self.slurm_job_results = self.__run_autocrop_on_all_books()

        # 2. Wait till all cropping has been finished
        # self.wait()

        print("Slurm Job Results")
        for index in range(len(self.slurm_job_results)):
            print("Result {0}: {1}".format(index, self.slurm_job_results[index]))

        print("Exiting QA_Autocrop.run")

        # 3. Output result stats for book or all books
        # self.output_stats()

    def __run_autocrop_on_all_books(self):

        print("Entering/exiting QA_Autocrop.__run_autocrop_on_all_books")

        return [ self.__run_autocrop_on_book(format_path(self.config[BOOK_DIRECTORY] + book_name)) \
            for book_name in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]) \
            if RESULTS_DIRECTORY != book_name ]

    def __run_autocrop_on_book(self, p_book_directory):

        print("Entering QA_Autocrop.__run_autocrop_on_book")

        # 1. Start a process to test autocropping methods on this book
        book_name = Path(p_book_directory).name

        # 2. Path for error output will be in the top level results directory
        error_path = "{0}results{1}".format(format_path(str(p_book_directory)), os.sep)

        # 3. Spin up a slurm job to crop with each possible cropping type on this book
        slurm_results = []
        autocrop_job_names = []
        for autocrop_type in AUTOCROP_TYPES:

            # A. Determine output path for cropped images and create it if it does not exist
            output_path = "{0}results{1}{2}{1}".format(format_path(str(p_book_directory)), os.sep, autocrop_type)
            if not os.path.exists(output_path):
                os.makedirs(output_path)             
        
            print("Creating slurm job for QA of cropping {0} with autocrop type {1}".format(book_name, autocrop_type))

            # B. Run auto_crop.py on the book with the given arguments

            # I. Save the job ID for outputting stats on autocrop runs afterward
            job_name = "{0}_{1}_{2}".format(book_name, autocrop_type, self.config[RUN_UUID])
            autocrop_job_names.append(job_name)

            # II. sbatch arguments
            sbatch_directives = {
                
                "-c": SBATCH_NUMBER_CPUS,
                "-J": job_name,
                "--mem-per-cpu": SBATCH_MEMORY_PER_CPU,
                "-o": "{0}slurm-{1}_{2}_{3}.out".format(self.config[OUTPUT_DIRECTORY], book_name, autocrop_type, self.config[RUN_UUID]),
                "-p": SBATCH_PARTITION,
                "-t": SBATCH_TIME
            }

            # III. autocrop.py arguments
            autocrop_args = [

                AUTOCROP_SCRIPT_LOCATION,
                "--path", str(p_book_directory),
                "--output_path", output_path,
                "--error_path", error_path,
                "--run_uuid", self.config[RUN_UUID],
                "--test"
            ]
            if CROPTYPE_THRESHOLD_BY_INSIDE == autocrop_type:
                autocrop_args.append("--threshold_by_inside")
            autocrop_args.append("*.tif")

            # IV. Build the sbatch call
            subprocess_args = ""
            for arg in sbatch_directives:
                subprocess_args += " {0} {1}".format(arg, sbatch_directives[arg])
            subprocess_args += " " + "qa_autocrop_new.sh"
            for arg in autocrop_args:
                subprocess_args += " " + arg
            subprocess_cmd = "sbatch " + subprocess_args

            # print("subprocess.run(sbatch {0}, capture_output=True, text=True, shell=True)".format(subprocess_args))
            print("subprocess.Popen({0} shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)".format(subprocess_cmd))

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
            # slurm_results.append(
            #     self.start_process(# command, args, description
            #         "sbatch",
            #         subprocess_args,
            #         "{0}_{1}_{2}".format(book_name, autocrop_type, self.config[RUN_UUID])
            #     )
            # )
        
        # 4. Build and run a dependent sbatch call for the output of stats on these autocrop runs for this book
        # sbatch_output_stats = "sbatch"
        # sbatch_output_directives = {

        #     "-c": "1",
        #     "-d": "afterany:{0}".format(":".join(autocrop_job_ids)),
        #     "--mem": "1g",
        #     "-o": "{0}slurm-output-{1}_{2}.out".format(self.config[OUTPUT_DIRECTORY], book_name, self.config[RUN_UUID]),
        #     "-p": "RM-shared",
        #     "-t": "12:00:00"
        # }
        # for arg in sbatch_output_directives:
        #     sbatch_output_stats += " {0} {1}".format(arg, sbatch_output_directives[arg])
        # sbatch_output_stats += " qa_autocrop_outputstats.sh"
        # # sbatch_output_stats += " --wrap=\"python3 {0}qa.py autocrop --single_book --output_stats --book_directory {1} --run_uuid {2}\"".format(format_path(os.getcwd()), book_name, self.config[RUN_UUID])

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
        # # subprocess.run(sbatch_output_stats, capture_output=True, text=True, shell=True))
        # # print("subprocess.run({0}, capture_output=True, text=True, shell=True)".format(sbatch_output_stats))

        print("Exiting QA_Autocrop.__run_autocrop_on_book")
        
        return slurm_results

    def output_stats(self):

        print("Entering QA_Autocrop.output_stats")

        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            self.__output_stats_on_book(self.config[BOOK_DIRECTORY])
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            self.__output_stats_on_all_books()

        print("Exiting QA_Autocrop.output_stats")

    def __output_stats_on_all_books(self):

        print("Entering QA_Autocrop.__output_stats_on_all_books")

        for book_name in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):

            # Skip results directory
            if RESULTS_DIRECTORY == book_name:
                continue

            # A. sbatch arguments
            sbatch_directives = {
                
                "-c": SBATCH_NUMBER_CPUS,
                "-J": "{0}_{1}".format(book_name, self.config[RUN_UUID]),
                "--mem-per-cpu": SBATCH_MEMORY_PER_CPU,
                "-o": "{0}slurm-output-{1}_{2}.out".format(self.config[OUTPUT_DIRECTORY], book_name, self.config[RUN_UUID]),
                "-p": SBATCH_PARTITION,
                "-t": SBATCH_TIME
            }

            # B. Build the sbatch call
            subprocess_args = ""
            for arg in sbatch_directives:
                subprocess_args += " {0} {1}".format(arg, sbatch_directives[arg])
            subprocess_args += " --wrap=\"python3 qa.py autocrop --single_book --output_stats --book_directory {0} --output_directory {1} --run_uuid {2}\"".format(
                self.config[BOOK_DIRECTORY] + book_name, self.config[OUTPUT_DIRECTORY], self.config[RUN_UUID])
            subprocess_cmd = "sbatch " + subprocess_args

            print("subprocess.Popen({0} shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)".format(subprocess_cmd))

            # C. Run sbatch and save results
            subprocess.Popen(
                subprocess_cmd,
                shell=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True
            )

        print("Exiting QA_Autocrop.__output_stats_on_all_books")

        # return [ self.__output_stats_on_book(format_path(self.config[BOOK_DIRECTORY] + book_name)) \
        #     for book_name in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]) \
        #     if RESULTS_DIRECTORY != book_name ]

    def __output_stats_on_book(self, p_book_directory):

        print("Entering QA_Autocrop.__output_stats_on_book")

        # 0. Output path
        output_folder = format_path(p_book_directory)

        csv_results = {}

        book_dir = p_book_directory
        book_name = os.path.basename(book_dir[0:len(p_book_directory)-1])
        csv_results[book_name] = { "original": {} }
        results_folder = "{0}results{1}".format(output_folder, os.sep)

        print("Book directory: " + book_dir)
        print("Book name: " + book_name)
        print("Results folder: " + results_folder)

        # 1. Output stats csv files for each cropping run on this book
        for autocrop_type in AUTOCROP_TYPES:

            print("Loop for " + autocrop_type)

            autocrop_type_subfolder = results_folder + autocrop_type

            print("autocrop_type_subfolder: " + autocrop_type_subfolder)

            # Check to make sure that the cropping run produced a directory of images
            # if not os.path.exists(autocrop_type_subfolder):
            #     print("Cropping for {0} using method '{1}' did not produce any images.".format(book_name, autocrop_type))
            #     print("No stats csv file for this cropping run will be output.")
            #     return    

            # A. Potential error file for this cropping run for this book
            error_filepath = "{0}results{1}error_{2}_{3}_{4}.txt".format(output_folder, os.sep,
                book_name, autocrop_type, self.config[RUN_UUID])
            error_lookup = {}
            if os.path.exists(error_filepath):
                error_lookup = self.read_error_file(error_filepath, "AUTOCROP")

            print("FINISHED READING ERROR LOOKUP TABLE")
            print("ERROR_LOOKUP:\n{0}".format(error_lookup))

            # B. Determine info about the original images

            # I. The number of original images
            csv_results[book_name]["original"]["file_count"] = len(get_items_in_dir(str(book_dir), ["files"]))
            csv_results[book_name]["original"]["images"] = {}

            # II. Gather stats on the original book images
            for image_filepath in Path(p_book_directory).glob("*.tif"):

                print("Loop for original image: {0}".format(image_filepath))

                try:
                    img = Image.open(image_filepath)
                except UnidentifiedImageError:
                    print("Unidentified image error for {0}".format(Path(image_filepath).name))
                    error_lookup[Path(image_filepath).name] = str(traceback.format_exc())
                    continue
                image_name = os.path.basename(image_filepath)

                # print("Image name: " + image_name)

                csv_results[book_name]["original"]["images"][image_name] = { "binarized_image": binarize_img(img)[0] }

                # print("Binarized image done")

                # Image area
                csv_results[book_name]["original"]["images"][image_name]["image_width"] = img.size[0]
                csv_results[book_name]["original"]["images"][image_name]["image_height"] = img.size[1]
                csv_results[book_name]["original"]["images"][image_name]["image_area"]  = img.size[0] * img.size[1]

                # print("Image area done")

                # N/A values
                csv_results[book_name]["original"]["images"][image_name]["area_diff_from_original"] = 0
                csv_results[book_name]["original"]["images"][image_name]["percent_area_diff_from_original"] = 0
                csv_results[book_name]["original"]["images"][image_name]["frobenius_norm_from_original"] = 0
                csv_results[book_name]["original"]["images"][image_name]["min_pct_dimension_difference"] = 0
                csv_results[book_name]["original"]["images"][image_name]["error"] = "N/A"

                # print("N/A values done")

            # C. Comparisons between originals and autocrop run

            csv_results[book_name][autocrop_type] = {}

            # I. The number of autocropped images
            csv_results[book_name][autocrop_type]["file_count"] = len(get_items_in_dir(autocrop_type_subfolder, ["files"]))
            csv_results[book_name][autocrop_type]["images"] = {}

            # print("Pre gather stats loop")

            # II. Gather stats on autocropped images and compare to original images
            for image_filepath in Path(autocrop_type_subfolder).glob("*.tif"):

                print("Loop for cropped image: {0}".format(image_filepath))

                try:
                    img = Image.open(image_filepath)
                except Exception as e:
                    print("Image opening exception for {0}".format(image_filepath))
                    error_lookup[Path(image_filepath).name] = str(traceback.format_exc())
                    continue
                image_name = os.path.basename(image_filepath)

                # a. Find second to last dash in cropped image filepath
                # original_image_name = image_name[image_name.rfind("-", 0, image_name.rfind("-")) + 1:]
                csv_results[book_name][autocrop_type]["images"][image_name] = {}

                # b. Image area comparison
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
                
                # c. Frobenius norm between original and autocropped images

                # i. Pad the autocropped image to the size of the original
                try:
                    new_image = Image.new(
                        img.mode,
                        (csv_results[book_name]["original"]["images"][image_name]["image_width"],
                        csv_results[book_name]["original"]["images"][image_name]["image_height"]),
                    ) 
                    new_image.paste(img, (0, 0))
                except:
                    print("ERROR: Problem creating image for comparison with cropped in __output_stats_for_book.")
                    print("Image: {0}".format(image_name))
                    error_lookup[Path(image_filepath).name] = str(traceback.format_exc())
                    continue

                # ii. Binarize the autocropped image
                autocrop_img_mtx = np.asarray(binarize_img(new_image)[0]).astype(int)

                # iii. Calculate the Frobenius norm between the two binarized images
                original_img_mtx = np.asarray(csv_results[book_name]["original"]["images"][image_name]["binarized_image"]).astype(int)
                diffed_img_mtx = np.subtract(original_img_mtx, autocrop_img_mtx)
                csv_results[book_name][autocrop_type]["images"][image_name]["frobenius_norm_from_original"] = np.linalg.norm(diffed_img_mtx, "fro")

                # d. All images found are likely not errored
                csv_results[book_name][autocrop_type]["images"][image_name]["error"] = "N/A"

            # III. Add in errored images with their errors
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

            # D. Output a csv file of these stats in the autocrop result folder
            for book_name in csv_results:

                results_folder = "{0}results{1}".format(output_folder, os.sep)
                stats_filepath = results_folder + "{0}_{1}_{2}.csv".format(STATS_FILE_PREFIX, autocrop_type, self.config[RUN_UUID])

                print("Outputting stats for {0} to {1}".format(book_name, stats_filepath))

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
        
        print("Exiting QA_Autocrop.__output_stats_on_book")

    def wait(self):

        print("Entering QA_Autocrop.wait with run type: {0}".format(self.config[RUN_TYPE]))

        if RUN_TYPE_SINGLE == self.config[RUN_TYPE]:
            self.__wait_for_autocrop_on_book(self.config[BOOK_DIRECTORY])
        elif RUN_TYPE_MULTI == self.config[RUN_TYPE]:
            self.__wait_for_autocrop_on_all_books()

        print("Exiting QA_Autocrop.wait")

    def __wait_for_autocrop_on_book(self, p_book_directory):

        print("Entering QA_Autocrop.__wait_for_autocrop_on_book with book directory: {0}".format(p_book_directory))

        while not self.is_method_finished(p_book_directory):
            time.sleep(5)
            continue

        print("Exiting QA_Autocrop.__wait_for_autocrop_on_book")
    
    def __wait_for_autocrop_on_all_books(self):

        print("Entering/exiting QA_Autocrop.__wait_for_autocrop_on_all_books")

        return all([self.__wait_for_autocrop_on_book(format_path(self.config[BOOK_DIRECTORY] + book_name)) \
                    for book_name in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]) \
                    if RESULTS_DIRECTORY != book_name])
