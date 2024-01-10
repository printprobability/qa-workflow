# Author: Jonathan Armoza
# Created: October 13, 2023
# Purpose: Classes and functions for Print & Probability QA scripts

# Imports

# Built-ins
import ast
import csv
import glob
import importlib
import inspect
import math
import os
import queue
import shutil
import subprocess
import sys
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

        print("Entering/exiting QA_Module.collate_logs")

        pass
    def collate_results(self):

        print("Entering/exiting QA_Module.collate_results")

        pass  

    def data_stats(self):

        print("Entering QA_Module.data_stats")

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

        print("Exiting QA_Module.data_stats")

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

def find_missing_images_le(p_book_directory, p_output_directory, p_single_or_multi):

    print("p_book_directory: " + p_book_directory)
    print("p_output_directory: " + p_output_directory)
    print("p_single_or_multi: " + p_single_or_multi)

    def find_missing_images_le_helper(p_single_book_directory):

        # Returns: 
        # 1. number of line tifs in directory if no line_df.csv else number of lines in line_df.csv,
        # 2. Missing image filepaths
        # 3. Whether or not line_df.csv was found in the 'lines_color' subdirectory

        lines_color_dir = format_path(p_single_book_directory + "lines_color")
        missing_image_filepaths = []
        num_lines = 0

        if not os.path.exists(lines_color_dir):
            print("Could not find 'lines_color' directory for {0}".format(p_single_book_directory))
            return 0, [], False
        elif not os.path.exists(f"{lines_color_dir}line_df.csv"):
            print("Could not find line_df.csv for {0}".format(p_single_book_directory))
            line_tifs = [ filename for filename in get_items_in_dir(lines_color_dir, ["files"]) if filename.endswith(".tif") ]
            return len(line_tifs), [], False

        with open(f"{lines_color_dir}line_df.csv", "r") as input_file:
            csv_reader = csv.DictReader(input_file)
            for row in csv_reader:
                num_lines += 1
                real_filename = row["file_name"][0:row["file_name"].rfind("-")] + row["file_name"][row["file_name"].rfind("-") + 1:] + ".tif"
                if not os.path.exists(lines_color_dir + real_filename):
                    missing_image_filepaths.append(lines_color_dir + real_filename)
        
        return num_lines, missing_image_filepaths, True

    book_dir = format_path(p_book_directory)
    output_dir = format_path(p_output_directory)

    missing_image_filepaths = []
    book_stats = {}
    if "multi" == p_single_or_multi:
        for directory in get_items_in_dir(book_dir, ["directories"]):

            before_count = len(missing_image_filepaths)
            print("Looking for missing images in {0}".format(directory), flush=True)

            num_lines, filepaths, found_linedf = find_missing_images_le_helper(format_path(book_dir + directory))
            missing_image_filepaths.extend(filepaths)

            print("Found {0} missing images.".format(len(missing_image_filepaths) - before_count), flush=True)

            book_stats[directory] = {

                "lines": num_lines,
                "missing_lines": len(filepaths),
                "line_df_exists": found_linedf
            }
    else:
        missing_image_filepaths = find_missing_images_le_helper(book_dir)

    run_uuid = get_unique_uuid(output_dir, "*.csv")

    with open("{0}missing_le_images_{1}.csv".format(output_dir, run_uuid), "w") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(["image_filepath"])
        for filepath in missing_image_filepaths:
            csv_writer.writerow([filepath])
    with open("{0}le_stats_{1}.csv".format(output_dir, run_uuid), "w") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(["book_directory","total_lines", "missing_lines", "line_df_exists"])
        for directory in book_stats:
            csv_writer.writerow([
                directory,
                book_stats[directory]["lines"],
                book_stats[directory]["missing_lines"],
                book_stats[directory]["line_df_exists"]
            ])
        
def format_path(original_path):
    '''Make sure given path ends with system folder separator'''
    return original_path if original_path.endswith(os.sep) else original_path + os.sep

def get_fastest_eynollah_images(p_book_directory, p_n):

    eynollah_run_time_key = "INFO:eynollah:Job done in "
    eynollah_command = "eynollah -m"
    image_times = {}

    # 1. Read slurm log files for images in each book directory
    book_directories = get_items_in_dir(p_book_directory, ["directories"])
    for book_directory in book_directories:

        book_directory_fullpath = format_path(os.path.join(p_book_directory, book_directory))
        for log_filepath in glob.glob(book_directory_fullpath + "*.out"):

            image_filename = None
            run_time = None

            with open(log_filepath, "r") as log_file:
                lines = log_file.readlines()
                for line in lines:

                    # A. Look for image filename
                    if line.startswith(eynollah_command):
                        image_filename = os.path.basename(line.split(" ")[4])

                    # B. Look for line that indicates the completion time of this eynollah run
                    if line.startswith(eynollah_run_time_key):
                        run_time = float(line.split(" ")[3].strip().strip("s"))

                    if image_filename and run_time:
                        image_times[image_filename] = {
                            "directory": book_directory_fullpath,
                            "log_filepath": log_filepath,
                            "run_time": run_time
                        }

    

    # 2. Determine lowest N eynollah completion times
    sorted_times = [(it_key, image_times[it_key]["run_time"]) for it_key in image_times]
    sorted_times = sorted(sorted_times, key=lambda image_info: image_info[1])

    time_range = [0,0]
    time_range[0] = min([item[1] for item in sorted_times])
    time_range[1] = max([item[1] for item in sorted_times])

    print("Number of images: {0}".format(len(image_times.items())))
    print(f"Lowest time: {time_range[0]}")
    print(f"Highest time: {time_range[1]}")

    print(f"{p_n} images with lowest times:")
    for index in range(int(p_n)):
        image_filename = sorted_times[index][0]
        print("=" * 80)
        print(f"\tImage: {image_filename}")
        print(f"\tTime: {sorted_times[index][1]}")
        print(f"\tDirectory: {image_times[image_filename]['directory']}")
        print(f"\tLog file: {image_times[image_filename]['log_filepath']}")

def get_eynollah_times(p_book_directory):

    import matplotlib.pyplot as plt
    import numpy as np    

    booklevel_stats = {}

    # 1. Read in book level start logs
    for slurm_log_file in glob.glob(p_book_directory + os.sep + "*.out"):

        with open(os.path.join(p_book_directory, slurm_log_file)) as log_file:
            log_lines = log_file.readlines()

        book_name = None

        # A. Save part 1 job ID from filename
        filename = os.path.basename(slurm_log_file)
        job_id = filename[filename.find("-") + 1:]
        job_id = job_id.strip(".out")

        for line in log_lines:

            # B. Save book name
            if line.startswith("BOOK NAME:"):
                book_name = line[line.find(":") + 2:].strip()
                booklevel_stats[book_name] = { "jobid_part1": job_id }
            
            # C. Get start time
            if line.startswith("Thu"):
                booklevel_stats[book_name]["book_starttime"] = line.split(" ")[3]

            # D. Get ID of dependent job (part 2)
            if line.startswith("sbatch --dependency=afterany:"):
                line_parts = line.split(" ")
                booklevel_stats[book_name]["jobid_part2"] = line_parts[1].split(":")[1].strip()
                
        # E. Get end time of part 2 job
        for name in booklevel_stats:
            part2_filename = f"slurm-{booklevel_stats[name]['jobid_part2']}.out"
            if os.path.exists(os.path.join(p_book_directory, part2_filename)):
                with open(os.path.join(p_book_directory, part2_filename), "r") as part2_logfile:
                    pass

        # F. Calculate and store book run time
        for name in booklevel_stats:
            # booklevel_stats[name]["book_endtime"] = 0
            # booklevel_stats[name]["book_runtime"] = booklevel_stats[name]["book_endtime"] - booklevel_stats[name]["book_starttime"]
            pass

    # 2. For each book, read page level logs
    for name in booklevel_stats:

        booklevel_stats[name]["page_seconds_list"] = []

        # A. Get runtime for each page
        for page_logfilepath in glob.glob(os.path.join(p_book_directory, name) + os.sep + "*.out"):
            
            with open(page_logfilepath, "r") as page_logfile:
                lines = page_logfile.readlines()
                for line in lines:
                    if line.startswith("INFO:eynollah:Job done in "):
                        line_parts = line.split("INFO:eynollah:Job done in ")
                        seconds = line_parts[1].strip()
                        seconds = float(seconds[0:len(seconds) - 3])
                        booklevel_stats[name]["page_seconds_list"].append(seconds)

    # 3. Calculate distribution for book run times and output data to file
    flattened_page_seconds = [seconds for name in booklevel_stats for seconds in booklevel_stats[name]["page_seconds_list"] ]
    print("Flattened seconds: {0}".format(flattened_page_seconds))

    # 4. Calculate distributions for page run times and output data to file
    plt.hist(flattened_page_seconds, color="lightgreen", ec="black", bins=10)
    plt.show()

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

def get_line_extraction_angles(p_search_directory):

    search_dir = format_path(p_search_directory)

    total_lines = 0

    for directory in get_items_in_dir(search_dir, ["directories"]):

        angles = []
        angle_buckets = { "0": 0, "90": 0, "other": 0 }
        angle_dist = {}
        image_angle_dict = {}
        for index in range(10):
            angle_dist[str(index * 10)] = 0

        lines_color_dir = format_path(search_dir + directory + os.sep + "lines_color")

        # print("lines_color_dir: {0}".format(lines_color_dir))

        if os.path.exists(lines_color_dir) and os.path.exists(lines_color_dir + "line_df.csv"):
            with open(lines_color_dir + "line_df.csv", "r") as le_file:
                csv_reader = csv.DictReader(le_file)
                for row in csv_reader:

                    total_lines += 1

                    angle_of_rotation = float(ast.literal_eval(row["rect"])[2])

                    real_filename = row["file_name"][0:row["file_name"].rfind("-")] + row["file_name"][row["file_name"].rfind("-") + 1:] + ".tif"
                    image_angle_dict[real_filename] = {
                        "angle": angle_of_rotation,
                        "path": lines_color_dir + real_filename,
                        "funny": False
                    }
                    angles.append(angle_of_rotation)

                    if math.isclose(angle_of_rotation, 0, abs_tol=1):
                        angle_buckets["0"] += 1
                    elif math.isclose(angle_of_rotation, 90, abs_tol=1):
                        angle_buckets["90"] += 1
                    else:
                        angle_buckets["other"] += 1

                        if angle_of_rotation > 10 and angle_of_rotation < 80:
                            image_angle_dict[real_filename]["funny"] = True

                    for bucket in angle_dist:
                        if angle_of_rotation >= float(bucket) and \
                           angle_of_rotation <= float(bucket) + 10:
                            angle_dist[bucket] += 1
        
    unique_angles = list(set(angles))

    print("Total lines: {0}".format(total_lines))

    # print("Unique angle count: {0}".format(len(unique_angles)))
    # print("Angle buckets:")
    # for angle in angle_buckets:
    #     print("Angle {0}: {1}".format(angle, angle_buckets[angle]))
    
    # print("Angle distribution: {0}".format(angle_dist))
    # print(angles)

    # with open(os.getcwd() + os.sep + "angle_bins.csv", "w") as output_file:
    #     csv_writer = csv.writer(output_file)
    #     csv_writer.writerow(["bucket", "count"])
    #     for bucket in angle_dist:
    #         csv_writer.writerow([bucket, angle_dist[bucket]])

    with open(os.getcwd() + os.sep + "funny_angled_lines.csv", "w") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(["filepath", "angle"])
        for filename in image_angle_dict:
            if image_angle_dict[filename]["funny"]:
                csv_writer.writerow([image_angle_dict[filename]["path"], image_angle_dict[filename]["angle"]])

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

def print_debug_header(p_header="", p_header_character="=", p_header_length=80):

    print("{0} {1}".format(p_header, p_header_character * (p_header_length - len(p_header) - 1)))

def read_error_file(p_error_filepath_with_wildcard, p_module_name):

    print("Entering read_error_file")
    print("p_error_filepath_with_wildcard: " + p_error_filepath_with_wildcard)
    print("p_module_name: " + p_module_name)
    
    error_lookup = {}
    error_filepath = ""
    
    # 0. Use first file that matches the given filepath with wildcard
    for filepath in glob.glob(p_error_filepath_with_wildcard):
        error_filepath = filepath
        break
    if "" == error_filepath:
        return {}
    
    print("Error filepath: {0}".format(error_filepath))

    # 1. Store errors from this file keyed by filename listed in each error
    with open(error_filepath, "r") as error_file:
        error_lines = error_file.readlines()

        print("Read error lines")

        begin_error = False
        image_filename = ""
        recording_traceback = False
        tb_lines = []
        for index in range(len(error_lines)):

            # print("Processing line: {0}".format(error_lines[index]))

            if f"BEGIN {p_module_name} FAILURE" in error_lines[index]:
                # print("Found BEGIN AUTOCROP FAILURE")
                begin_error = True
                continue
            if begin_error and "FILE:" in error_lines[index]:
                # print("begin_error is true and found FILE")
                image_filename = Path(error_lines[index].split("FILE: ")[1].strip()).name
                # print("image_filename: {0}".format(image_filename))

                if image_filename not in error_lookup:
                    error_lookup[image_filename] = []
                continue
            if begin_error and "ERROR:" in error_lines[index]:
                # print("Found ERROR")
                recording_traceback = True
                continue
            if recording_traceback:
                if "END" in error_lines[index]:
                    # print("Found error END")
                    error_lookup[image_filename].append(tb_lines.copy())
                    # print("error_lookup[{0}]:\n{1}".format(image_filename, error_lookup[image_filename]))
                    begin_error = False
                    image_filename = ""
                    recording_traceback = False
                    tb_lines = []
                else:
                    # print("Appending error line")
                    tb_lines.append("\"" + error_lines[index].strip() + "\"")

    print("Exiting read_error_file")
    
    return error_lookup

def scale_image(p_image_filepath, p_scale_factor, p_scale_tag="scaled"):

    # 1. Load the image into memory
    try:
        image = Image.open(p_image_filepath)
    except UnidentifiedImageError:
        print(f"Tried to scale '{p_image_filepath}' and received UnidentifiedImageError")
        return

    # 2. Scale the image by the given factor
    image.thumbnail((image.size[0] * float(p_scale_factor), image.size[1] * float * p_scale_factor))

    # 3. Save image to disk under filename with given tag
    new_image_name = f"{Path(p_image_filepath).stem}_{p_scale_tag}{Path(p_image_filepath).suffix}"
    image.save(os.path.join(Path(p_image_filepath).parent, new_image_name)) 

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


def main(p_args):

    # Make sure utility function name is given and the arguments it needs
    if len(p_args) < 2 or len(p_args[2:]) != len(inspect.getfullargspec(globals()[p_args[1]])[0]):
        print("qa_utilities.py usage: ")
        print("python[3] qa_utilities.py <utility function name> <exact args list for function or none if it has no args>")
        return

    # Call utility function with the given arguments
    globals()[p_args[1]](*p_args[2:])

if "__main__" == __name__:
    main(sys.argv)