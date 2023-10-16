# Author: Jonathan Armoza
# Created: October 11, 2023
# Purpose: Master script for the quality assurance of the Print & Probability
# book processing and ingestion pipeline. Reads in a config yaml file for input
# and output folders, and a sequential list of commands for the QA workflow.

# Imports

# Built-ins
import argparse
import glob
import json
import os
import yaml
import subprocess
from pathlib import Path

# Custom
from qa_utilities import *


# Globals

qa_config = {}
slurm_job_results = None
VALID_COMMANDS = ["clear_output", "run_qa", "collate_results"]
VALID_QA_TYPES = ["autocrop"]
DEFAULT_OUTPUT_DIRECTORY = "output"

def clear_output():

    if input("Are you sure you want to clear output directory {0}? (y/n)".format(qa_config["OUTPUT_DIRECTORY"])) == "y":
        print("Output directory cleared.")
    else:
        print("Output directory not cleared.")

def collate_results():

    for book_directory in get_items_in_dir(format_path(qa_config["BOOK_DIRECTORY"]), ["directories"]):

        # Skip the output directory
        if os.path.basename(qa_config["OUTPUT_DIRECTORY"]) == book_directory:
            continue

        if os.path.exists("{0}{1}results{1}".format(book_directory, os.sep)):
            pass       

def run_qa():

    if "autocrop" == qa_config["QA_TYPE"]:

        if "single" == qa_config["RUN_TYPE"]:
            slurm_job_results = qa_autocrop_on_book(os.path.basename(qa_config["BOOK_DIRECTORY"]))
        else:
            slurm_job_results = qa_autocrop_on_all_books()

def qa_autocrop_on_all_books():

    return [ qa_autocrop_on_book(book_name) \
        for book_name in get_items_in_dir(format_path(qa_config["BOOK_DIRECTORY"]), ["directories"]) \
        if output_folder != book_name ]

def qa_autocrop_on_book(p_book_name):

    print("Creating slurm job for QA of cropping " + directory)

    return subprocess.Popen([
        "sbatch",
        "-o", "{0}slurm-{1}.out".format(qa_config["OUTPUT_DIRECTORY"]),  
        "qa_autocrop.sh",
        format_path(qa_config["BOOK_DIRECTORY"] + directory),
        qa_config["COMMANDS"]
    ])

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
        "--single-book",
        action="store_true",
        help="QA the cropping of a single book. NOTE: Requires a '--book_directory' and an '--output_directory'")

    # 2. Run argparser over args given for this script run
    args = parser.parse_args()

    # 3. Determine if arg requirements are met
    success = True
    if not args.qa_function:
        print("Must specify QA function type. Current options: 'autocrop'")
        success = False
    if not(args.single_book or args.config_file) or \
        (args.single_book and args.config_file):
        print("QA script must specify single book or book directory via config file.")
        success = False
    elif args.single_book:
        if not (args.book_directory and args.output_directory):
            print("Single book runs must specify the book directory.")
            print('Example: python qa.py --single_book --book_directory "/my/example/book/dir/"')
            success = False
        if not os.path.exists(args.book_directory):
            print("Book directory: {0} does not exist.".format(args.book_directory))
        elif not os.path.isdir(args.book_directory):
            print("Book directory: {0} is not a directory.".format(args.book_directory))
            success = False
    elif args.config_file:
        if not os.path.exists(args.config_file):
            print("Config file: {0} does not exist.".format(args.config_file))
            success = False
        elif not os.path.isfile(args.config_file):
            print("Config file: {0} is not a file.".format(args.config_file))
            success = False
    if args.output_directory:
        # stolast_sep = args.output_directory[:args.output_directory.rfind(os.sep)].rfind(os.sep)
        # output_parent_dir = args.output_directory[0:stolast_sep]
        output_parent_directory = Path(args.output_directory).parent
        if not os.path.exists(output_parent_directory):
            print("Output directory's parent: {0} does not exist.".format(output_parent_directory))
            success = False
        elif not os.path.isdir(output_parent_directory):
            print("Output directory's parent: {0} is not a directory.".format(output_parent_directory))
            success = False

    return args, success

def save_config(p_args):

    success = True
    config_required_fields = ["BOOK_DIRECTORY"]

    # 1. Save default config values
    qa_config["COMMANDS"]=["run_qa"]
    qa_config["OUTPUT_DIRECTORY"] = "{0}{1}{2}{1}".format(p_args.book_directory, os.sep, DEFAULT_OUTPUT_DIRECTORY)

    # 2. Save optional config values if given
    if p_args.output_directory:
        qa_config["OUTPUT_DIRECTORY"] = p_args.output_directory

    # 3. Save mandatory config values
    if p_args.single_book:

        qa_config["RUN_TYPE"] = "single"
        qa_config["BOOK_DIRECTORY"] = p_args.book_directory
        qa_config["QA_TYPE"] = p_args.qa_function

        # A. Check for TIF images in the book directory
        items = get_items_in_dir(qa_config["BOOK_DIRECTORY"], return_types=["files"])
        found_tif = False
        for item in items:
            if item.lowercase().endswith(".tif"):
                found_tif = True
        if not found_tif:
            print("Could not find any tif images in the book directory: {0}.".format(qa_config["BOOK_DIRECTORY"]))
            success = False
    else:

        qa_config["RUN_TYPE"] = "multi"
        qa_config["QA_TYPE"] = p_args.qa_function

        # A. Read in config yaml file and save its fields
        with open(p_args.config_file, "r") as config_file:
            config_yaml = yaml.safe_load(config_file)
        if "BOOK_DIRECTORY" in config_yaml:
            qa_config["BOOK_DIRECTORY"] = config_yaml["BOOK_DIRECTORY"]
        if "COMMANDS" in config_yaml:
            qa_config["COMMANDS"] = config_yaml["COMMANDS"]
        if "OUTPUT_DIRECTORY" in config_yaml:
            qa_config["OUTPUT_DIRECTORY"] = config_yaml["OUTPUT_DIRECTORY"]

        # B. Check contents of config file
        if not all(cmd in config_yaml.keys() for cmd in config_required_fields):
            print("Config files require at least the 'BOOK_DIRECTORY' key.")
            success = False
        for cmd in qa_config["COMMANDS"]:
            if cmd not in VALID_COMMANDS:
                print("{0} is an invalid command. Valid commands: {1}".format(cmd, VALID_COMMANDS))
                success = False
        if qa_config["QA_TYPE"] not in VALID_QA_TYPES:
            print("{0} is an invalid qa type. Valid qa types: {1}".format(qa_config["QA_TYPE"], VALID_QA_TYPES))
            success = False
            
    # 4. Check config elements common to both single and multi-book runs
    if not os.path.exists(qa_config["BOOK_DIRECTORY"]):
        print("Book directory: {0} does not exist.".format(qa_config["BOOK_DIRECTORY"]))
        success = False
    elif not os.path.isdir(qa_config["BOOK_DIRECTORY"]):
        print("Book directory: {0} is not a directory.".format(qa_config["BOOK_DIRECTORY"]))
        success = False
    output_parent_directory = Path(qa_config["OUTPUT_DIRECTORY"]).parent
    if not os.path.exists(output_parent_directory):
        print("Output directory's parent: {0} does not exist.".format(output_parent_directory))
        success = False
    elif not os.path.isdir(output_parent_directory):
        print("Output directory: {0} is not a directory.".format(output_parent_directory))
        success = False            

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