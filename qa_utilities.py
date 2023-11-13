# Author: Jonathan Armoza
# Created: October 13, 2023
# Purpose: Classes and functions for Print & Probability QA scripts

# Imports

# Built-ins
import csv
import glob
import importlib
import os
import queue
import shutil
import subprocess
import _thread
import uuid
from pathlib import Path

# Third party
from PIL import Image
from PIL import UnidentifiedImageError

# Custom
from qa_constants import *

# Classes

# QA module base class
class QA_Module:

    def __init__(self, p_config):
        
        print("Entering QA_Module.__init__")
        
        self.config = p_config
        self.process_queue = QAProcessWaiter()

        print("Exiting QA_Module.__init__")

    def call_command(self, p_command_name):

        print("Entering QA_Module.call_command")
        
        getattr(self, p_command_name)()
        # self.wait()

        print("Exiting QA_Module.call_command")

    def is_method_finished(self, p_book_directory):

        print("Entering/exiting QA_Module.is_method_finished")

        return False

    # QA methods (listable in "COMMANDS" in config)

    def archive(self):

        print("Entering QA_Module.archive")

        self.archive_logs()
        self.archive_results()

        print("Exiting QA_Module.archive")

    def archive_logs(self):

        print("Entering/exiting QA_Module.archive_logs")

        pass    
    def archive_results(self):

        print("Entering/exiting QA_Module.archive_results")

        pass

    def clear(self):

        print("Entering QA_Module.clear")

        self.clear_logs()
        self.clear_results()

        print("Exiting QA_Module.clear")

    def clear_logs(self):

        print("Entering/exiting QA_Module.clear_logs")

        pass
    def clear_results(self):

        print("Entering/exiting QA_Module.clear_results")

        pass

    def collate(self):

        print("Entering QA_Module.collate")

        self.collate_logs()
        self.collate_results()
        self.collate_errors()

        print("Exiting QA_Module.collate")

    def collate_errors(self):

        print("Entering/exiting QA_Module.collate_errors")

        pass

    def collate_logs(self):

        print("Entering/exiting collate_logs")

        pass
    def collate_results(self):

        print("Entering/exiting collate_results")

        pass  

    def data_stats(self):

        # 1. Calculate stats for book and page images
        book_stats = {}
        for book_directory in get_items_in_dir(self.config[BOOK_DIRECTORY], ["directories"]):
            
            book_stats[book_directory] = {
                "images": {},
                "num_pages": 0
            }

            for image_filepath in glob.glob(self.config[BOOK_DIRECTORY] + book_directory + os.sep + "*.tif"):
                
                width, height, file_size = get_image_stats(image_filepath)
                image_name = Path(image_filepath).name

                book_stats[book_directory]["num_pages"] += 1
                book_stats[book_directory]["images"][image_name] = {}
                book_stats[book_directory]["images"][image_name]["width"] = width
                book_stats[book_directory]["images"][image_name]["height"] = height
                book_stats[book_directory]["images"][image_name]["file_size"] = file_size

        # 2. Write out results to data stats file in output directory
        with open(self.config[OUTPUT_DIRECTORY] + "data_stats_{0}.csv".format(self.config[RUN_UUID]), "w") as stats_file:

            csv_writer = csv.writer(stats_file)
            csv_writer.writerow([
                "image_filename",
                "num_pages_in_book",
                "file_size_bytes",
                "width_pixels",
                "height_pixels"
            ])

            for book in book_stats:
                for image_name in book_stats[book]["images"]:
                    csv_writer.writerow([
                        image_name,
                        book_stats[book]["num_pages"],
                        book_stats[book]["images"][image_name]["file_size"],
                        book_stats[book]["images"][image_name]["width"],
                        book_stats[book]["images"][image_name]["height"],
                    ])


    def run(self):

        print("Entering/exiting QA_Module.run")

        pass

    # Process queue methods
    def is_process_finished(self, p_description):

        print("Entering/exiting QA_Module.is_process_finished")

        return not self.process_queue.process_is_in_queue(p_description)
    def start_process(self, p_command, p_args, p_description):

        print("Entering QA_Module.start_process")

        self.process_queue.start_process(QAProcess(p_command, p_args, p_description))

        print("Exiting QA_Module.start_process")
    def wait(self):

        print("Entering QA_Module.wait")

        self.process_queue.wait_till_all_finished()

        print("Exiting QA_Module.wait")

class QAProcess:

    def __init__(self, p_command, p_args, p_description):

        self.m_command = p_command
        self.m_args = p_args
        self.m_description = p_description
        
        self.m_popen_handle = None

    def open(self):

        self.m_popen_handle = subprocess.Popen([self.m_command, *self.m_args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        # self.m_popen_handle = subprocess.run("{0} {1}".format(self.m_command, *self.m_args), capture_output=True, text=True, shell=True)

        return self.m_popen_handle

    @property
    def handle(self):
        return self.m_popen_handle
    
    @property
    def command(self):
        return self.m_command
    
    @property
    def args(self):
        return self.m_args
    
    @property
    def description(self):
        return self.m_description

class QAProcessWaiter:

    def __init__(self):

        self.m_processes = []
        self.m_queue = queue.Queue()

    def __process_waiter(self, p_process_handle, p_description):

        try:
            p_process_handle.wait()
        finally:
            self.m_queue.put((
                p_description,
                p_process_handle.returncode
            ))

    def __remove_process(self, p_description):

        found_process = False
        found_index = -1
        for index in range(len(self.m_processes)):
            if p_description == self.m_processes[index][0].description:
                found_index = index
                break
        
        if -1 != found_index:
            del self.m_processes[found_index]

    def process_is_in_queue(self, p_description):

        for index in range(len(self.m_processes)):
            if p_description == self.m_processes[index][0].description:
                return True
        return False

    def start_process(self, p_qa_process):

        if self.process_is_in_queue(p_qa_process.description):
            raise Exception("A QA process with description '{0}' is already in queue.\n" + \
                            "QA processes must have unique descriptions.")

        # 1. Save a reference to the process and open it
        process_handle = p_qa_process.open()
        self.m_processes.append((p_qa_process, process_handle))

        # 2. Begin a new thread to wait on this process's finish
        _thread.start_new_thread(
            self.__process_waiter,
            (p_qa_process.handle, p_qa_process.description)
        )

        return process_handle

    def wait_till_all_finished(self):

        # Wait for all processes on the queue to finish
        while len(self.m_processes) > 0:

            # 1. Wait for a process on the queue to finish
            description, return_code = self.m_queue.get()

            # 2. Once done, remove it from the process list
            self.__remove_process(description)

            print("Job {0} ended with return code: {1}".format(description, return_code))

# Functions

def copy_data_directory(p_src_directory, p_dest_directory):

    directories = get_items_in_dir(p_src_directory, ["directories"])
    for dir in directories:
        new_dir = p_dest_directory + dir + os.sep
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for src_filepath in glob.glob(p_src_directory + dir + os.sep + "*.tif"):
            filename = Path(src_filepath).name
            shutil.copyfile(src_filepath, new_dir + filename)    

def directory_has_files_of_type(p_book_directory, p_file_tag):

    items = get_items_in_dir(format_path(p_book_directory), return_types=["files"])
    found_tif = False
    for item in items:
        if item.lower().endswith(p_file_tag):
            found_tif = True
    return found_tif

def find_errors(p_errors_to_look_for, p_directory, p_filesearch_str_w_wildcard):

    files_containing_errors = { error_string:[] for error_string in p_errors_to_look_for }

    for filepath in glob.glob(p_directory + p_filesearch_str_w_wildcard):
        with open(filepath, "r") as log_file:
            lines = log_file.readlines()
            for line in lines:
                for error in p_errors_to_look_for:
                    if error in line:
                        files_containing_errors[error].append(Path(filepath).name)

    for error in p_errors_to_look_for:
        print("{0}:".format(error))
        for filename in files_containing_errors[error]:
            print("\t{0}".format(filename))
        print("=" * 80)

def format_path(original_path):
    '''Make sure given path ends with system folder separator'''
    return original_path if original_path.endswith(os.sep) else original_path + os.sep

def get_uniquer_error_line(p_error):

    error_search_string = "Error:"
    key_line = ""
    formatted_error = p_error

    # 0. Split strings with endlines into a list
    if "\n" in formatted_error:
        formatted_error = formatted_error.split("\n")

    # 1. Find the error line
    if isinstance(formatted_error, list):
        for line in formatted_error:
            if error_search_string in line:
                key_line = line
                break
    elif isinstance(formatted_error, str):
        key_line = formatted_error

    # 2. Remove paths in the error line
    if os.sep in key_line:
        key_line = key_line[0:key_line.find(os.sep)]

    return key_line.strip()

def get_image_stats(p_image_filepath):

    try:
        img = Image.open(p_image_filepath)
    except UnidentifiedImageError:
        print("Unidentified image error for {0}".format(Path(p_image_filepath).name))
        return "N/A", "N/A", "N/A"

    width = img.size[0]
    height = img.size[1]
    file_size = os.stat(p_image_filepath).st_size

    return width, height, file_size

def get_items_in_dir(path, return_types=[]):
    '''Get all items in given path of type "directories" and/or "files"'''

    formatted_path = format_path(path)
    directory_contents = os.listdir(formatted_path)
    returned_contents = []

    for dI in directory_contents:
        if "directories" in return_types:
            if os.path.isdir(os.path.join(formatted_path, dI)):
                returned_contents.append(dI)
        if "files" in return_types:
            if not os.path.isdir(os.path.join(formatted_path, dI)):
                returned_contents.append(dI)

    return returned_contents

def get_unique_uuid(p_search_directory, p_search_string):

    # 1. Get a random UUID
    new_uuid = uuid.uuid4()

    # 2. Make sure no file in the search directory matching the given search string has this UUID
    filepaths = glob.glob(p_search_directory + p_search_string)
    for index in range(len(filepaths)):

        # A. Get a new UUID if the current one is detected in a filename
        if str(new_uuid) in os.path.basename(filepaths[index]):
            new_uuid = uuid.uuid4()
            index = 0

    return str(new_uuid)

def makedirs(p_location):

    if os.path.exists(p_location):
        shutil.rmtree(p_location, ignore_errors=True)
    os.makedirs(p_location)

def str_to_class(module_name, class_name):

    """Return a class instance from a string reference"""
    # Taken from https://stackoverflow.com/a/24674853/3831152

    try:
        module_ = importlib.import_module(module_name)
        try:
            # class_ = getattr(module_, class_name)()
            class_ = getattr(module_, class_name)
        except AttributeError:
            print("Class {0} does not exist".format(class_name))
    except ImportError:
        print("Module {0} does not exist".format(module_name))
            
    return class_ or None

def traceback_to_str(p_traceback):

    '''Makes sure given traceback from exception is in string form'''
    return " ".join(p_traceback) if isinstance(p_traceback, list) else p_traceback

def wait_while_exists(p_path):

    '''For after uses of unlink or rmtree; waits until passed in item still exists on disk'''
    # Taken from https://stackoverflow.com/questions/21505313/is-there-a-foolproof-way-to-give-the-system-enough-time-to-delete-a-folder-befor

    while os.path.exists(p_path):
        pass