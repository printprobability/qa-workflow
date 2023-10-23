# Author: Jonathan Armoza
# Created: October 11, 2023
# Purpose: Master script for the quality assurance of the Print & Probability
# book processing and ingestion pipeline. Reads in a config yaml file for input
# and output folders, and a sequential list of commands for the QA workflow.

# Imports

# Built-ins
import argparse
import csv
import glob
import os
import yaml
import subprocess
from datetime import datetime
from pathlib import Path

# Custom
from qa_utilities import *


# Globals

qa_config = {}
slurm_job_results = None

# Constants
VALID_COMMANDS = ["clear_output", "run_qa", "collate_results"]
VALID_QA_TYPES = ["autocrop"]
DEFAULT_OUTPUT_DIRECTORY = "{0}{1}output{1}".format(os.getcwd(), os.sep)


def clear_output():

    if input("Are you sure you want to clear output directory {0}? (y/n)".format(qa_config["OUTPUT_DIRECTORY"])) == "y":
        print("Output directory cleared.", flush=True)
    else:
        print("Output directory not cleared.", flush=True)

def collate_results():

    if "single" == qa_config["RUN_TYPE"]:
        collate_results_on_book(qa_config["BOOK_DIRECTORY"] + "results" + os.sep)
    else:
        for book_directory in get_items_in_dir(format_path(qa_config["BOOK_DIRECTORY"]), ["directories"]):

            full_bookpath = format_path(qa_config["BOOK_DIRECTORY"] + book_directory)

            # Skip the QA log output directory if it exists in the book directory
            if Path(qa_config["OUTPUT_DIRECTORY"]).name == Path(full_bookpath):
                continue
            
            # Write out a merged results file for this book in its results directory
            collate_results_on_book(full_bookpath + "results" + os.sep)

def collate_results_on_book(p_results_directory):

    # 1. Get two most recent csv files for autocropping in the results directory (ignoring other collation csvs)
    csv_filepaths = [(filepath, os.path.getctime(filepath)) \
        for filepath in glob.glob(p_results_directory + "*.csv") if "merged_" not in filepath]
    if len(csv_filepaths) < 2:
        raise Exception("Less than two csv files in the results directory: {0}".format(p_results_directory))
    sorted_csv_filepaths = sorted(csv_filepaths, key=lambda filepath: filepath[1], reverse=True)
    results_filepath1, results_filepath2 = sorted_csv_filepaths[0][0],sorted_csv_filepaths[1][0]

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
    with open(p_results_directory + "merged_results_{0}.csv".format(datetime.now().timestamp()), "w") as output_file:
        csv_writer = csv.writer(output_file)

        csv_writer.writerow([
            "book_name",
            "total_page_count",
            "autocrop_type",
            "image_name",
            "image_width",
            "image_height",
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
                row["image_area"],
                row["area_diff_from_original"],
                row["percent_area_diff_from_original"],
                row["frobenius_norm_from_original"]
            ])

def run_qa():

    if "autocrop" == qa_config["QA_TYPE"]:

        if "single" == qa_config["RUN_TYPE"]:
            slurm_job_results = qa_autocrop_on_book(qa_config["BOOK_DIRECTORY"])
        else:
            slurm_job_results = qa_autocrop_on_all_books()

    print("Slurm Job Results")
    for index in range(len(slurm_job_results)):
        print("Result {0}: {1}".format(index, slurm_job_results[index]))

def qa_autocrop_on_all_books():

    return [ qa_autocrop_on_book(format_path(qa_config["BOOK_DIRECTORY"] + book_name)) \
        for book_name in get_items_in_dir(qa_config["BOOK_DIRECTORY"], ["directories"]) \
        if Path(qa_config["OUTPUT_DIRECTORY"]).name != book_name ]

def qa_autocrop_on_book(p_book_directory): 

    # 1. Start a process to test autocropping methods on this book
    book_name = Path(p_book_directory).name
    print("Creating slurm job for QA of cropping " + book_name, flush=True)

    # A. Build subprocess arguments for sbatch call
    sbatch_args = {

        "-c": "2",
        # "-J": "{0}-{1}".format(book_name, datetime.now().timestamp()),
        "--mem-per-cpu": "1999mb",
        "-t": "06:00:00",
        "-o": "{0}slurm-{1}-{2}.out".format(qa_config["OUTPUT_DIRECTORY"], book_name, datetime.now().timestamp()),
        "-p": "RM-shared"
    }
    subprocess_cmd = "sbatch"
    for arg in sbatch_args:
        subprocess_cmd += " {0} {1}".format(arg, sbatch_args[arg])
    subprocess_cmd += " {0}{1}qa_autocrop.sh {2}".format(os.getcwd(), os.sep, p_book_directory)

    # subprocess_cmd = "sbatch {0}{1}qa_autocrop.sh".format(os.getcwd(), os.sep)
    print("subprocess.run({0}, capture_output=True, text=True, shell=True)".format(subprocess_cmd), flush=True)

    return [subprocess.run(subprocess_cmd, capture_output=True, text=True, shell=True)]

# Main script

def handle_args():

    # 1. Set up argparser with args for script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "qa_function",
        help="Part of the pipeline you want to QA. Current options: 'autocrop'")
    parser.add_argument(
        "--config_file",
        help="Path to a yaml configuration file for your QA run")
    parser.add_argument(
        "--book_directory",
        help="Directory containing the images of one or more books")
    parser.add_argument(
        "--output_directory",
        help="Directory where QA output should go. Default output is ")
    parser.add_argument(
        "--single_book",
        action="store_true",
        help="QA the cropping of a single book. NOTE: Requires a '--book_directory' and an '--output_directory'")

    # 2. Run argparser over args given for this script run
    args = parser.parse_args()

    # 3. Determine if arg requirements are met
    success = True
    if not args.qa_function:
        print("Must specify QA function type. Current options: 'autocrop'", flush=True)
        success = False
    elif args.single_book:
        if not args.book_directory:
            print("Single book runs must specify the book directory.", flush=True)
            print('Example: python qa.py --single_book --book_directory "/my/example/book/dir/"', flush=True)
            success = False
        if not os.path.exists(args.book_directory):
            print("Book directory: {0} does not exist.".format(args.book_directory), flush=True)
            success = False
        elif not os.path.isdir(args.book_directory):
            print("Book directory: {0} is not a directory.".format(args.book_directory), flush=True)
            success = False
    elif args.config_file:
        if not os.path.exists(args.config_file):
            print("Config file: {0} does not exist.".format(args.config_file), flush=True)
            success = False
        elif not os.path.isfile(args.config_file):
            print("Config file: {0} is not a file.".format(args.config_file), flush=True)
            success = False
    if args.output_directory:
        output_parent_directory = Path(args.output_directory).parent
        if not os.path.exists(output_parent_directory):
            print("Output directory's parent: {0} does not exist.".format(output_parent_directory), flush=True)
            success = False
        elif not os.path.isdir(output_parent_directory):
            print("Output directory's parent: {0} is not a directory.".format(output_parent_directory), flush=True)
            success = False

    return args, success

def save_config(p_args):

    success = True
    config_required_fields = ["BOOK_DIRECTORY"]

    # 1. Save default config values
    qa_config["COMMANDS"]=["run_qa"]
    qa_config["OUTPUT_DIRECTORY"] = DEFAULT_OUTPUT_DIRECTORY

    # 2. Save optional config values if given
    if p_args.output_directory:
        qa_config["OUTPUT_DIRECTORY"] = format_path(p_args.output_directory)

    # 3. Save mandatory config values
    if p_args.single_book:

        qa_config["QA_TYPE"] = p_args.qa_function
        qa_config["RUN_TYPE"] = "single"
        qa_config["BOOK_DIRECTORY"] = format_path(p_args.book_directory)

        # A. Check for TIF images in the book directory
        if not directory_has_files_of_type(qa_config["BOOK_DIRECTORY"], ".tif"):
            print("Could not find any tif images in the book directory: {0}.".format(qa_config["BOOK_DIRECTORY"]), flush=True)
            success = False
    else:

        qa_config["RUN_TYPE"] = "multi"
        qa_config["QA_TYPE"] = p_args.qa_function

        # A. Read in config yaml file and save its fields
        with open(p_args.config_file, "r") as config_file:
            config_yaml = yaml.safe_load(config_file)
        if "BOOK_DIRECTORY" in config_yaml:
            qa_config["BOOK_DIRECTORY"] = format_path(config_yaml["BOOK_DIRECTORY"])
        if "COMMANDS" in config_yaml:
            qa_config["COMMANDS"] = config_yaml["COMMANDS"]
        if "OUTPUT_DIRECTORY" in config_yaml:
            qa_config["OUTPUT_DIRECTORY"] = format_path(config_yaml["OUTPUT_DIRECTORY"])
        # NOTE: RUN_TYPE can also be 'single' here to indicate a single book run that is using a config file
        if "RUN_TYPE" in config_yaml:
            qa_config["RUN_TYPE"] = config_yaml["RUN_TYPE"]

        # B. Check contents of config file
        if not all(cmd in config_yaml.keys() for cmd in config_required_fields):
            print("Config files require at least the 'BOOK_DIRECTORY' key.", flush=True)
            success = False
        for cmd in qa_config["COMMANDS"]:
            if cmd not in VALID_COMMANDS:
                print("{0} is an invalid command. Valid commands: {1}".format(cmd, VALID_COMMANDS), flush=True)
                success = False
        if qa_config["QA_TYPE"] not in VALID_QA_TYPES:
            print("{0} is an invalid qa type. Valid qa types: {1}".format(qa_config["QA_TYPE"], VALID_QA_TYPES), flush=True)
            success = False
            
    # 4. Check config elements common to both single and multi-book runs
    if not os.path.exists(qa_config["BOOK_DIRECTORY"]):
        print("Book directory: {0} does not exist.".format(qa_config["BOOK_DIRECTORY"]), flush=True)
        success = False
    elif not os.path.isdir(qa_config["BOOK_DIRECTORY"]):
        print("Book directory: {0} is not a directory.".format(qa_config["BOOK_DIRECTORY"]), flush=True)
        success = False
    output_parent_directory = Path(qa_config["OUTPUT_DIRECTORY"]).parent
    if not os.path.exists(output_parent_directory):
        print("Output directory's parent: {0} does not exist.".format(output_parent_directory), flush=True)
        success = False
    elif not os.path.isdir(output_parent_directory):
        print("Output directory: {0} is not a directory.".format(output_parent_directory), flush=True)
        success = False

    print("QA CONFIG")
    for key in qa_config:
        print("{0}: {1}".format(key, qa_config[key]), flush=True)           

    return success

def main():

    # 1. Handle args given to this script
    args, success = handle_args()
    if not success:
        exit()
    
    # 2. Save args for QA run
    if not save_config(args):
        exit()

    # 3. Run QA commands from loaded config
    for cmd in qa_config["COMMANDS"]:
        globals()[cmd]()


if "__main__" == __name__:
    main()