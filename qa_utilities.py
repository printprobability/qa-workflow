# Author: Jonathan Armoza
# Created: October 13, 2023
# Purpose: Classes and functions for Print & Probability QA scripts

# Imports

# Built-ins
import glob
import importlib
import os
import uuid


# Classes

# QA module base class
class QA_Module:

    def __init__(self, p_config):
        pass

    def archive(self):
        self.archive_logs()
        self.archive_results()

    def archive_logs(self):
        pass    
    def archive_results(self):
        pass

    def clear(self):
        self.clear_logs()
        self.clear_results()
    def clear_logs(self):
        pass
    def clear_results(self):
        pass

    def collate(self):
        self.collate_logs()
        self.collate_results()
    def collate_logs(self):
        pass
    def collate_results(self):
        pass    

    def run(self):
        pass  

# Functions

def directory_has_files_of_type(p_book_directory, p_file_tag):

    items = get_items_in_dir(format_path(p_book_directory), return_types=["files"])
    found_tif = False
    for item in items:
        if item.lower().endswith(p_file_tag):
            found_tif = True
    return found_tif

def format_path(original_path):
    '''Make sure given path ends with system folder separator'''
    return original_path if original_path.endswith(os.sep) else original_path + os.sep

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