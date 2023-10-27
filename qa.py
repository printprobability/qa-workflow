# Author: Jonathan Armoza
# Created: October 11, 2023
# Purpose: Master script for the quality assurance of the Print & Probability
# book processing and ingestion pipeline. Reads in a config yaml file for input
# and output folders, and a sequential list of commands for the QA workflow.

# Imports

# Built-ins
import argparse
import os
from pathlib import Path

# Third party
import yaml

# Custom
from qa_constants import *
from qa_utilities import *


# Globals

qa_config = {}
slurm_job_results = None


# Master log creation

def create_error_log():

    # Look at each log in output directory that contains UUID 
    # qa_config["OUTPUT_DIRECTORY"]
    pass


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
        print("Must specify QA function type. Current options: {0}".format(", ".join(QA_TYPE_CLASSES.keys()),
              flush=True)
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

def run_commands():

    # 0. Load up the QA class module and instantiate a copy of it
    module_name = QA_TYPE_CLASSES[qa_config[QA_TYPE]][0]
    class_name = QA_TYPE_CLASSES[qa_config[QA_TYPE]][1]
    qa_module = str_to_class(module_name, class_name)(qa_config)

    # 1. Run QA commands in the sequence listed in the loaded config
    for cmd in qa_config[COMMANDS]:
        getattr(qa_module, cmd)()
        
def save_config(p_args):

    success = True
    config_required_fields = [BOOK_DIRECTORY]

    # 1. Save default config values
    qa_config[COMMANDS]=[COMMANDS_RUN]
    qa_config[OUTPUT_DIRECTORY] = DEFAULT_OUTPUT_DIRECTORY

    # 2. Save optional config values if given
    if p_args.output_directory:
        qa_config[OUTPUT_DIRECTORY] = format_path(p_args.output_directory)

    # 3. Save mandatory config values
    if p_args.single_book:

        qa_config[QA_TYPE] = p_args.qa_function
        qa_config[RUN_TYPE] = RUN_TYPE_SINGLE
        qa_config[BOOK_DIRECTORY] = format_path(p_args.book_directory)

        # A. Check for TIF images in the book directory
        if not directory_has_files_of_type(qa_config[BOOK_DIRECTORY], ".tif"):
            print("Could not find any tif images in the book directory: {0}.".format(qa_config[BOOK_DIRECTORY]), flush=True)
            success = False
    else:

        qa_config[RUN_TYPE] = RUN_TYPE_MULTI
        qa_config[QA_TYPE] = p_args.qa_function

        # A. Read in config yaml file and save its fields
        with open(p_args.config_file, "r") as config_file:
            config_yaml = yaml.safe_load(config_file)
        if BOOK_DIRECTORY in config_yaml:
            qa_config[BOOK_DIRECTORY] = format_path(config_yaml[BOOK_DIRECTORY])
        if COMMANDS in config_yaml:
            qa_config[COMMANDS] = config_yaml[COMMANDS]
        if OUTPUT_DIRECTORY in config_yaml:
            qa_config[OUTPUT_DIRECTORY] = format_path(config_yaml[OUTPUT_DIRECTORY])
        # NOTE: RUN_TYPE can also be 'single' here to indicate a single book run that is using a config file
        if RUN_TYPE in config_yaml:
            qa_config[RUN_TYPE] = config_yaml[RUN_TYPE]

        # B. Check contents of config file
        if not all(cmd in config_yaml.keys() for cmd in config_required_fields):
            print("Config files require at least the 'BOOK_DIRECTORY' key.", flush=True)
            success = False
        for cmd in qa_config[COMMANDS]:
            if cmd not in VALID_COMMANDS:
                print("{0} is an invalid command. Valid commands: {1}".format(cmd, VALID_COMMANDS), flush=True)
                success = False
        if qa_config[QA_TYPE] not in VALID_QA_TYPES:
            print("{0} is an invalid qa type. Valid qa types: {1}".format(qa_config[QA_TYPE], VALID_QA_TYPES), flush=True)
            success = False
            
    # 4. Check config elements common to both single and multi-book runs
    if not os.path.exists(qa_config[BOOK_DIRECTORY]):
        print("Book directory: {0} does not exist.".format(qa_config[BOOK_DIRECTORY]), flush=True)
        success = False
    elif not os.path.isdir(qa_config[BOOK_DIRECTORY]):
        print("Book directory: {0} is not a directory.".format(qa_config[BOOK_DIRECTORY]), flush=True)
        success = False
    output_parent_directory = Path(qa_config[OUTPUT_DIRECTORY]).parent
    if not os.path.exists(output_parent_directory):
        print("Output directory's parent: {0} does not exist.".format(output_parent_directory), flush=True)
        success = False
    elif not os.path.isdir(output_parent_directory):
        print("Output directory: {0} is not a directory.".format(output_parent_directory), flush=True)
        success = False

    # 5. Get a unique UUID for this run
    qa_config[RUN_UUID] = get_unique_uuid(qa_config[OUTPUT_DIRECTORY], MERGED_RESULTS_FILENAME_PREFIX + "*")

    return success

def main():

    # 1. Handle args given to this script
    args, success = handle_args()
    if not success:
        exit()
    
    # 2. Save args for QA run
    if not save_config(args):
        exit()

    # 3. Run requested commands for this QA module
    run_commands()


if "__main__" == __name__:
    main()