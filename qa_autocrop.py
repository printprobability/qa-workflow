"""
Tests the autocropper on the test set of books created by create_autocrop_test_dir.py.
"""
import csv
import numpy as np
import os
import subprocess
from datetime import datetime
from pathlib import Path
from PIL import Image
from prepare_alignment_input_csv import *
from qa_utilities import *


# Globals

# Constants

CROPTYPE_THRESHOLD_BY_INSIDE = "threshold_by_inside"
CROPTYPE_NON_THRESHOLD_BY_INSIDE = "non_threshold_by_inside"

AUTOCROP_SCRIPT_LOCATION = "..{0}auto_crop.py".format(os.sep)


# Main script functions

def run_autocrop(args):

    """ Call auto_crop.py for the given book """
    
    autocrop_type = CROPTYPE_THRESHOLD_BY_INSIDE if args.threshold_by_inside else CROPTYPE_NON_THRESHOLD_BY_INSIDE
            
    # 1. Determine output path for cropped images and create it if it does not exist
    output_path = "{0}results{1}{2}{1}".format(format_path(str(args.book_directory)), os.sep, autocrop_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 2. Run auto_crop.py on the book with the given arguments
    subprocess_args = ["python3", AUTOCROP_SCRIPT_LOCATION, "--path", str(args.book_directory), "--output_path", output_path]
    if CROPTYPE_THRESHOLD_BY_INSIDE == autocrop_type:
        subprocess_args.append("--threshold_by_inside")
    subprocess_args.append("*.tif")
    subprocess.run(subprocess_args)


def output_stats(args):

    # 0. Output path
    output_folder = format_path(args.book_directory)

    csv_results = {}

    book_dir = args.book_directory
    book_name = os.path.basename(book_dir[0:len(book_dir)-1])
    csv_results[book_name] = { "original": {} }
    results_folder = "{0}results{1}".format(output_folder, os.sep)

    print("Book dir: {0}".format(book_dir))
    print("Book name: {0}".format(book_name))

    # 1. Determine info about the original images

    # A. The number of original images
    csv_results[book_name]["original"]["file_count"] = len(get_items_in_dir(str(book_dir), ["files"]))
    csv_results[book_name]["original"]["images"] = {}

    # B. Gather stats on the original book images
    for image_filepath in Path(args.book_directory).glob("*.tif"):

        img = Image.open(image_filepath)
        image_name = os.path.basename(image_filepath)
        csv_results[book_name]["original"]["images"][image_name] = { "binarized_image": binarize_img(img)[0] }

        # Image area
        csv_results[book_name]["original"]["images"][image_name]["image_width"] = img.size[0]
        csv_results[book_name]["original"]["images"][image_name]["image_height"] = img.size[1]
        csv_results[book_name]["original"]["images"][image_name]["image_area"]  = img.size[0] * img.size[1]

        # N/A values
        csv_results[book_name]["original"]["images"][image_name]["area_diff_from_original"] = 0
        csv_results[book_name]["original"]["images"][image_name]["percent_area_diff_from_original"] = 0
        csv_results[book_name]["original"]["images"][image_name]["frobenius_norm_from_original"] = 0

    # 2. Comparisons between originals and autocrop run
    # for autocrop_type in autocrop_types:
    autocrop_type = CROPTYPE_THRESHOLD_BY_INSIDE if args.threshold_by_inside else CROPTYPE_NON_THRESHOLD_BY_INSIDE

    autocrop_type_subfolder = results_folder + autocrop_type

    csv_results[book_name][autocrop_type] = {}

    # A. The number of autocropped images
    csv_results[book_name][autocrop_type]["file_count"] = len(get_items_in_dir(autocrop_type_subfolder, ["files"]))
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
        csv_results[book_name][autocrop_type]["images"][image_name]["percent_area_diff_from_original"] = 100.0f * \
            (float(csv_results[book_name]["original"]["images"][image_name]["image_area"]) / \
             float(csv_results[book_name]["original"]["images"][original_image_name]["image_area"]))

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

        results_folder = "{0}results{1}".format(output_folder, os.sep)
        stats_filepath = results_folder + "autocrop_results_{0}.csv".format(datetime.now().timestamp())

        with open(stats_filepath, "w") as output_file:

            csv_writer = csv.writer(output_file)

            csv_writer.writerow(["book_name", "total_page_count", "autocrop_type", "image_name", "image_area", "area_diff_from_original", "frobenius_norm_from_original"])

            for autocrop_type in csv_results[book_name]:

                for image_name in csv_results[book_name][autocrop_type]["images"]:

                    csv_writer.writerow([book_name, csv_results[book_name][autocrop_type]["file_count"], autocrop_type, image_name,
                                         csv_results[book_name][autocrop_type]["images"][image_name]["image_width"],
                                         csv_results[book_name][autocrop_type]["images"][image_name]["image_height"],
                                         csv_results[book_name][autocrop_type]["images"][image_name]["image_area"],
                                         csv_results[book_name][autocrop_type]["images"][image_name]["area_diff_from_original"],
                                         csv_results[book_name][autocrop_type]["images"][image_name]["frobenius_norm_from_original"]])

def parse_args():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("book_directory", help="Directory containing the images of one book copied using the create_autocrop_test_dir.py script")
    parser.add_argument("--threshold_by_inside", help="Computes binary threshold off inner chunk of page.", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    run_autocrop(args)
    output_stats(args)