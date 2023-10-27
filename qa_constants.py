# Author: Jonathan Armoza
# Created: October 26, 2023
# Purpose: Constants for the 'Print & Probability' QA workflow for book processing

# Directories and filenames
ARCHIVE_DIRECTORY = "archive"
DEFAULT_OUTPUT_DIRECTORY = "{0}{1}output{1}".format(os.getcwd(), os.sep)
RESULTS_DIRECTORY = "results"

MASTER_LOG_FILENAME_PREFIX = "qa_autocrop_slurm"
MERGED_RESULTS_FILENAME_PREFIX = "all_results_merged"

# Yaml config keys
BOOK_DIRECTORY = "BOOK_DIRECTORY"
COMMANDS = "COMMANDS"
OUTPUT_DIRECTORY = "OUTPUT_DIRECTORY"
QA_TYPE = "QA_TYPE"
RUN_TYPE = "RUN_TYPE"
RUN_UUID = "RUN_UUID"

# Yaml config values
COMMAND_ARCHIVE = "archive"
COMMAND_ARCHIVE_LOGS = "archive_logs"
COMMAND_ARCHIVE_RESULTS = "archive_results"
COMMAND_CLEAR = "clear"
COMMAND_CLEAR_LOGS = "clear_logs"
COMMAND_CLEAR_RESULTS = "clear_results"
COMMAND_RUN = "run"
COMMAND_COLLATE = "collate"
COMMAND_COLLATE_LOGS = "collate_logs"
COMMAND_COLLATE_RESULTS = "collate_results"
VALID_COMMANDS = [
    COMMAND_ARCHIVE,
    COMMAND_ARCHIVE_LOGS,
    COMMAND_ARCHIVE_RESULTS,
    COMMAND_CLEAR,
    COMMAND_CLEAR_LOGS,
    COMMAND_CLEAR_RESULTS,
    COMMAND_RUN,
    COMMAND_COLLATE,
    COMMAND_COLLATE_LOGS,
    COMMAND_COLLATE_RESULTS
]
QA_TYPE_AUTOCROP = "autocrop"
QA_TYPE_LINE_EXTRACTION = "line_extraction"
QA_TYPE_CLASSES = {
    QA_TYPE_AUTOCROP: ["qa_autocrop", "QA_AutoCrop"],
    QA_TYPE_LINE_EXTRACTION: ["qa_line_extraction", "QA_LineExtraction"]
}
VALID_QA_TYPES = [
    QA_TYPE_AUTOCROP,
    QA_TYPE_LINE_EXTRACTION
]
RUN_TYPE_MULTI = "multi"
RUN_TYPE_SINGLE = "single"
VALID_RUN_TYPES = [
    RUN_TYPE_MULTI, RUN_TYPE_SINGLE
]