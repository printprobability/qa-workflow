"""
Tests the autocropper on the test set of books created by create_autocrop_test_dir.py.
"""
import csv
import numpy as np
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image
from prepare_alignment_input_csv import *

# Utility functions

def format_path(original_path):

    return original_path if original_path.endswith(os.sep) else original_path + os.sep

# return_types - "subfolders" and/or "files"
def typed_listdir(path, return_types=[]):

    directory_contents = os.listdir(path)
    returned_contents = []

    for dI in directory_contents:

        if "subfolders" in return_types:
            if os.path.isdir(os.path.join(path, dI)):
                returned_contents.append(dI)
        elif "files" in return_types:
            if not os.path.isdir(os.path.join(path, dI)):
                returned_contents.append(dI)

    return returned_contents


# Main script functions

def run_autocrop(args):

    """ Call auto_crop.py for the given book """
    
    autocrop_type = "threshold_by_inside" if args.threshold_by_inside else "non_threshold_by_inside"
    book_dir = args.autocrop_test_dir
    
    # books = ["/ocean/projects/hum160002p/shared/books/test_autocrop/newcomb_R233270_uk_4_sermonpreachedjuly1676"]
    # for book_dir in books:
    # for book_dir in Path(format_path(args.autocrop_test_dir)).glob('*'):
    # for autocrop_type in autocrop_types:
        
    # Determine output path for cropped images
    output_path = "{0}results{1}".format(format_path(str(book_dir)), os.sep)
    output_path += "threshold_by_inside" if "threshold_by_inside" == autocrop_type else "non_threshold_by_inside"
    output_path += os.sep
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Run and test auto_crop.py on each book using subprocess
    # print(f"Running auto_crop.py on {book_dir}")
    subprocess_args = ["python3", "../auto_crop.py", "--path", str(book_dir), "--output_path", output_path]
    if "threshold_by_inside" == autocrop_type:
        subprocess_args.append("--threshold_by_inside")
    subprocess_args.append("*.tif")
    subprocess.run(subprocess_args)


def output_stats(args):

    # 0. Output paths
    output_folder = format_path(args.autocrop_test_dir)

    csv_results = {}

    # books = ["/ocean/projects/hum160002p/shared/books/test_autocrop/newcomb_R233270_uk_4_sermonpreachedjuly1676"]
    # for book_dir in books:
    # for book_dir in Path(args.autocrop_test_dir).glob('*'):

    book_dir = args.autocrop_test_dir
    book_name = os.path.basename(book_dir[0:len(book_dir)-1])
    csv_results[book_name] = { "original": {} }
    # results_folder = "{0}{1}{2}results{2}".format(output_folder, book_name, os.sep)
    results_folder = "{0}results{1}".format(output_folder, os.sep)

    print("Book dir: {0}".format(book_dir))
    print("Book name: {0}".format(book_name))
    # print("Results folder: {0}".format(results_folder))

    # 1. Determine info about the original images

    # A. The number of original images
    csv_results[book_name]["original"]["file_count"] = len(typed_listdir(str(book_dir), ["files"]))
    csv_results[book_name]["original"]["images"] = {}

    # B. Gather stats on the original book images
    # for image_filepath in Path(args.autocrop_test_dir + os.sep + book_name).glob("*.tif"):
    for image_filepath in Path(args.autocrop_test_dir).glob("*.tif"):

        img = Image.open(image_filepath)
        image_name = os.path.basename(image_filepath)
        csv_results[book_name]["original"]["images"][image_name] = { "binarized_image": binarize_img(img)[0] }

        # Image area
        csv_results[book_name]["original"]["images"][image_name]["image_width"] = img.size[0]
        csv_results[book_name]["original"]["images"][image_name]["image_height"] = img.size[1]
        csv_results[book_name]["original"]["images"][image_name]["image_area"]  = img.size[0] * img.size[1]

        # N/A values
        csv_results[book_name]["original"]["images"][image_name]["area_diff_from_original"] = 0
        csv_results[book_name]["original"]["images"][image_name]["frobenius_norm_from_original"] = 0

    # print("AUTOCROP TYPES: {0}".format(autocrop_types))

    # 2. Comparisons between originals and autocrop run
    # for autocrop_type in autocrop_types:
    autocrop_type = "threshold_by_inside" if args.threshold_by_inside else "non_threshold_by_inside"

    # print("Autocrop {0} comparison".format(autocrop_type))
    autocrop_type_subfolder = results_folder + autocrop_type

    csv_results[book_name][autocrop_type] = {}

    # A. The number of autocropped images
    csv_results[book_name][autocrop_type]["file_count"] = len(typed_listdir(autocrop_type_subfolder, ["files"]))
    csv_results[book_name][autocrop_type]["images"] = {} 

    # B. Gather stats on autocropped images and compare to original images
    for image_filepath in Path(autocrop_type_subfolder).glob("*.tif"):

        img = Image.open(image_filepath)
        image_name = os.path.basename(image_filepath)

        # I. Find second to last dash in cropped image filepath
        original_image_name = image_name[image_name.rfind("-", 0, image_name.rfind("-")) + 1:]
        csv_results[book_name][autocrop_type]["images"][image_name] = {}

        # II. Image area comparison
        csv_results[book_name][autocrop_type]["images"][image_name]["image_width"] = img.size[0]
        csv_results[book_name][autocrop_type]["images"][image_name]["image_height"] = img.size[1]
        csv_results[book_name][autocrop_type]["images"][image_name]["image_area"]  = img.size[0] * img.size[1]
        csv_results[book_name][autocrop_type]["images"][image_name]["area_diff_from_original"] = \
            csv_results[book_name]["original"]["images"][original_image_name]["image_area"] - \
            csv_results[book_name][autocrop_type]["images"][image_name]["image_area"]

        # III. Frobenius norm between original and autocropped images

        # a. Pad the autocropped image to the size of the original
        new_image = Image.new(
            img.mode,
            (csv_results[book_name]["original"]["images"][original_image_name]["image_width"],
            csv_results[book_name]["original"]["images"][original_image_name]["image_height"]),
        ) 
        new_image.paste(img, (0, 0))

        # b. Binarize the autocropped image
        autocrop_img_mtx = np.asarray(binarize_img(new_image)[0]).astype(int)

        # c. Calculate the Frobenius norm between the two binarized images
        original_img_mtx = np.asarray(csv_results[book_name]["original"]["images"][original_image_name]["binarized_image"]).astype(int)
        diffed_img_mtx = np.subtract(original_img_mtx, autocrop_img_mtx)
        csv_results[book_name][autocrop_type]["images"][image_name]["frobenius_norm_from_original"] = np.linalg.norm(diffed_img_mtx, "fro")

    # 3. Output a csv file of these stats in the autocrop result folder
    for book_name in csv_results:

        # results_folder = "{0}{1}{2}results{2}".format(output_folder, book_name, os.sep)
        results_folder = "{0}results{1}".format(output_folder, os.sep)
        stats_filepath = results_folder + "autocrop_results_{0}.csv".format(datetime.now().timestamp())

        # print("Writing to {0} for book {1}".format(stats_filepath, book_name))

        with open(stats_filepath, "w") as output_file:

            csv_writer = csv.writer(output_file)

            csv_writer.writerow(["book_name", "total_page_count", "autocrop_type", "image_name", "image_area", "area_diff_from_original", "frobenius_norm_from_original"])

            # print("Attempting to write autocropped images rows with csv_results[book_name].keys(): {0}".format(csv_results[book_name].keys()))

            for autocrop_type in csv_results[book_name]:
                # print("Loop 1 for autocrop_type".format(autocrop_type))
                for image_name in csv_results[book_name][autocrop_type]["images"]:
                    # print("Loop 2 for images: {0}".format(csv_results[book_name][autocrop_type]["images"]))
                    # print("Image: {0}".format(image_name))

                    # print("Writing row: {0}".format([book_name, csv_results[book_name][autocrop_type]["file_count"], autocrop_type, image_name,
                    #                      csv_results[book_name][autocrop_type]["images"][image_name]["image_area"],
                    #                      csv_results[book_name][autocrop_type]["images"][image_name]["area_diff_from_original"],
                    #                      csv_results[book_name][autocrop_type]["images"][image_name]["frobenius_norm_from_original"]]))

                    csv_writer.writerow([book_name, csv_results[book_name][autocrop_type]["file_count"], autocrop_type, image_name,
                                         csv_results[book_name][autocrop_type]["images"][image_name]["image_area"],
                                         csv_results[book_name][autocrop_type]["images"][image_name]["area_diff_from_original"],
                                         csv_results[book_name][autocrop_type]["images"][image_name]["frobenius_norm_from_original"]])


def parse_args():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("autocrop_test_dir", help="Directory containing the images of one book copied using the create_autocrop_test_dir.py script")
    parser.add_argument("--threshold_by_inside", help="Computes binary threshold off inner chunk of page.", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    run_autocrop(args)
    output_stats(args)
