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
from qa_autocrop import QA_Autocrop
from qa_constants import *
from qa_line_extraction import QA_LineExtraction
from qa_utilities import *


# Globals

qa_config = {}
slurm_job_results = None


# Main script

def handle_args():

    # 1. Set up argparser with args for script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "qa_function",
        help="Part of the pipeline you want to QA. Current options: 'autocrop' and 'line_extraction'")
    parser.add_argument(
        "--collate",
        action="store_true",
        help="Gather all results from cropping runs over books listed in config book directory")
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
        "--output_stats",
        action="store_true",
        help="Command to just output a csv file containing stats on a completed QA run")
    parser.add_argument(
        "--qa_subtype",
        help="Subtype of requested QA process (e.g. for line extraction: 'watershed', 'eynollah', etc."
    )
    parser.add_argument(
        "--single_book",
        action="store_true",
        help="QA the cropping of a single book. NOTE: Requires a '--book_directory' and an '--output_directory'")
    parser.add_argument(
        "--run_uuid",
        help="Optional UUID to continue usage of it from previous QA run")

    # 2. Run argparser over args given for this script run
    args = parser.parse_args()

    # 3. Determine if arg requirements are met
    success = True
    if not args.qa_function:
        print("Must specify QA function type. Current options: {0}".format(", ".join(QA_TYPE_CLASSES.keys())),
              flush=True)
        success = False
    elif args.single_book:
        if not args.book_directory:
            print("Single book runs must specify the book directory.")
            print('Example: python qa.py --single_book --book_directory "/my/example/book/dir/"')
            success = False
        if not os.path.exists(args.book_directory):
            print("Book directory: {0} does not exist.".format(args.book_directory))
            success = False
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
        output_parent_directory = Path(args.output_directory).parent
        if not os.path.exists(output_parent_directory):
            print("Output directory's parent: {0} does not exist.".format(output_parent_directory))
            success = False
        elif not os.path.isdir(output_parent_directory):
            print("Output directory's parent: {0} is not a directory.".format(output_parent_directory))
            success = False
    if args.qa_function:
        if QA_TYPE_LINE_EXTRACTION == args.qa_function:
            if not args.qa_subtype:
                print("A subtype for line extraction must be specified.")
                print("Current options: eynollah, watershed")
                success = False

    return args, success

def run_commands(p_args):

    # Special case to call results collation functionality - done when all results have completed
    if p_args.collate:
        qa_module.call_command("collate")
        return

    # 0. Load up the QA class module from config and instantiate a copy of it
    module_name, class_name = QA_TYPE_CLASSES[qa_config[QA_TYPE]]
    qa_module = str_to_class(module_name, class_name)(qa_config)

    # 1. Run QA commands in the sequence listed in the loaded config
    for cmd in qa_config[COMMANDS]:
        qa_module.call_command(cmd)
        
def save_config(p_args):

    success = True
    config_required_fields = [BOOK_DIRECTORY]

    # 1. Save default config values
    qa_config[COMMANDS]=[COMMAND_RUN]
    qa_config[OUTPUT_DIRECTORY] = DEFAULT_OUTPUT_DIRECTORY

    # 2. Save optional config values if given
    if p_args.output_directory:
        qa_config[OUTPUT_DIRECTORY] = format_path(p_args.output_directory)
    if p_args.output_stats:
        qa_config[COMMANDS] = [COMMAND_OUTPUT_STATS]

    # 3. Save mandatory config values
    if p_args.single_book:

        qa_config[QA_TYPE] = p_args.qa_function
        if p_args.qa_subtype:
            qa_config[QA_SUBTYPE] = p_args.qa_subtype
        qa_config[RUN_TYPE] = RUN_TYPE_SINGLE
        qa_config[BOOK_DIRECTORY] = format_path(p_args.book_directory)

        # A. Check for TIF images in the book directory
        if not directory_has_files_of_type(qa_config[BOOK_DIRECTORY], ".tif"):
            print("Could not find any tif images in the book directory: {0}.".format(qa_config[BOOK_DIRECTORY]))
            success = False

        # B. Make sure subtype is specified for this process if required
        if QA_TYPE_LINE_EXTRACTION == qa_config[QA_TYPE]:
            if not p_args.qa_subtype:
                print("A subtype for line extraction must be specified.")
                print("Current options: eynollah, watershed")
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
        if QA_SUBTYPE in config_yaml:
            qa_config[QA_SUBTYPE] = config_yaml[QA_SUBTYPE]

        # B. Check contents of config file
        if not all(cmd in config_yaml.keys() for cmd in config_required_fields):
            print("Config files require at least the 'BOOK_DIRECTORY' key.")
            success = False
        for cmd in qa_config[COMMANDS]:
            if cmd not in VALID_COMMANDS:
                print("{0} is an invalid command. Valid commands: {1}".format(cmd, VALID_COMMANDS))
                success = False
        if qa_config[QA_TYPE] not in VALID_QA_TYPES:
            print("{0} is an invalid qa type. Valid qa types: {1}".format(qa_config[QA_TYPE], VALID_QA_TYPES))
            success = False
        if QA_TYPE_LINE_EXTRACTION == qa_config[QA_TYPE] and QA_SUBTYPE not in qa_config:
            print("A subtype for line extraction must be specified.")
            print("Current options: eynollah, watershed")
            success = False

    # 4. Check config elements common to both single and multi-book runs
    if not os.path.exists(qa_config[BOOK_DIRECTORY]):
        print("Book directory: {0} does not exist.".format(qa_config[BOOK_DIRECTORY]))
        success = False
    elif not os.path.isdir(qa_config[BOOK_DIRECTORY]):
        print("Book directory: {0} is not a directory.".format(qa_config[BOOK_DIRECTORY]))
        success = False
    output_parent_directory = Path(qa_config[OUTPUT_DIRECTORY]).parent
    if not os.path.exists(output_parent_directory):
        print("Output directory's parent: {0} does not exist.".format(output_parent_directory))
        success = False
    elif not os.path.isdir(output_parent_directory):
        print("Output directory: {0} is not a directory.".format(output_parent_directory))
        success = False

    # 5. Get a unique UUID for this run, of previous UUID if given as argument
    if p_args.run_uuid:
        qa_config[RUN_UUID] = p_args.run_uuid
    elif RUN_UUID in config_yaml:
        qa_config[RUN_UUID] = config_yaml[RUN_UUID]
    else:
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
    run_commands(args)


if "__main__" == __name__:
    main()