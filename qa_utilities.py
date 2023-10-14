# Author: Jonathan Armoza
# Created: October 13, 2023
# Purpose: Utility functions for Print & Probability QA scripts

# Imports

# Built-ins
import os


# Functions

def format_path(original_path):
    '''Make sure given path ends with system folder separator'''
    return original_path if original_path.endswith(os.sep) else original_path + os.sep

def get_items_in_dir(path, return_types=[]):
    '''Get all items in given path of type "directories" and/or "files"'''

    directory_contents = os.listdir(path)
    returned_contents = []

    for dI in directory_contents:
        if "directories" in return_types:
            if os.path.isdir(os.path.join(path, dI)):
                returned_contents.append(dI)
        if "files" in return_types:
            if not os.path.isdir(os.path.join(path, dI)):
                returned_contents.append(dI)

    return returned_contents