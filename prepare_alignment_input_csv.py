"""
Author: Nikolai
Date: Apr 22, 2021
Script to loop through every color page, locally binarize the page, 
record the local character binarization thresholds based on the average 
threshold in the character bbox, and dump this and other relevant 
alignment input info to a csv.
"""
import csv
import io
import tarfile
from PIL import Image
import io
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import sys
from skimage.filters import threshold_sauvola


# GHT code from: https://github.com/jonbarron/hist_thresh/blob/master/experiments.ipynb  (not currently used)
# A fast numpy reference implementation of GHT, as per
# "A Generalization of Otsu's Method and Minimum Error Thresholding"
# Jonathan T. Barron, ECCV, 2020

csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
clip = lambda z: np.maximum(1e-30, z)


def preliminaries(n, x):
    """Some math that is shared across multiple algorithms."""
    assert np.all(n >= 0)
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    assert np.all(x[1:] >= x[:-1])
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x**2) - w0 * mu0**2
    d1 = dsum(n * x**2) - w1 * mu1**2
    return x, w0, w1, p0, p1, mu0, mu1, d0, d1


def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5, prelim=None):
    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert omega >= 0 and omega <= 1
    x, w0, w1, p0, p1, _, _, d0, d1 = prelim or preliminaries(n, x)
    v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa *      omega)  * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
    return argmax(x, f0 + f1), f0 + f1


def im2hist(im, zero_extents=False):
    # Convert an image to grayscale, bin it, and optionally zero out the first and last bins.
    max_val = np.iinfo(im.dtype).max
    x = np.arange(max_val+1)
    e = np.arange(-0.5, max_val+1.5)
    assert len(im.shape) in [2, 3]
    im_bw = np.amax(im[...,:3], -1) if len(im.shape) == 3 else im
    n = np.histogram(im_bw, e)[0]
    if zero_extents:
        n[0] = 0
        n[-1] = 0
    return n, x, im_bw


def binarize_img_ght(im):
    """ Returns a binarized PIL Image when provided a non-binarized PIL Image
    Using GHT
    :param im: PIL Image
    """
    im = np.array(im)
    # Precompute a histogram and some integrals.
    n, x, im_bw = im2hist(im)
    prelim = preliminaries(n, x)

    default_nu = np.sum(n)
    default_tau = np.sqrt(1/12)
    default_kappa = np.sum(n)
    default_omega = 0.5

    _nu = default_nu
    _tau = default_tau
    _kappa = default_kappa
    _omega = default_omega

    t, score = GHT(n, x, _nu, _tau, _kappa, _omega, prelim)
    return Image.fromarray(im_bw > t)


def binarize_img(im, window_size=25):
    """ Returns a binarized PIL Image when provided a non-binarized PIL Image
    Using Sauvola local adaptive thresholding
    :param im: PIL Image

    :returns: bin_img (of size HxW), thresholds (of size HxW) 
    """
    im = np.array(im)
    n, x, im_bw = im2hist(im)
    t = threshold_sauvola(im_bw, window_size=window_size)
    return Image.fromarray(im_bw > t), t


def extract_char_bboxes_by_page_from_json(json_dict):
    bboxes_by_page = defaultdict(list)
    # split out characters by page
    json_dict_by_page = defaultdict(list)
    for char_dict in json_dict['chars']:
        filename = char_dict['filename']
        right_bound_idx = filename.rfind('_page')
        pagenum = filename[filename[:right_bound_idx].rfind('-') + 1: right_bound_idx]
        #pagenum = Path(filename).name.split('-')[-1].split('_')[0]  # NOTE: doesn't work because of the '-' character found during OCR! 
        try:
            test = int(pagenum)
        except ValueError as e:
            print(pagenum)
            print(e)

        # APPEND char_dict to list of char_dicts on the page
        json_dict_by_page[pagenum].append(char_dict)
    
    # go thru the chars on each page and get the page image and all the char bboxes
    for pagenum, chars_on_page in sorted(json_dict_by_page.items(), key=lambda x: x[0]):
        for char_dict in chars_on_page:
            bboxes_by_page[pagenum].append(
                ([
                    char_dict['x_start_withpad'], 
                    char_dict['y_start_withpad'],
                    char_dict['x_end_withpad'],
                    char_dict['y_end_withpad']
                 ],
                 char_dict["filename"],
                 char_dict['logprob']
                )
            )
    return bboxes_by_page


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepares a CSV file with necessary info for alignment')
    parser.add_argument('--pages_color_root', help='Directory containing page images for the book')
    parser.add_argument('--book_char_images_tar', help='Path to book\'s char images tar file (probably in shared/char_images3')
    parser.add_argument('--json_output_root', help='Path to json_output')
    parser.add_argument('--csv_outfile', help='Path to csv output file for book')
    args = parser.parse_args()
    # Paths defined here: 
    json_output_root = Path(args.json_output_root)  # Path('/trunk/nvog/print-probability/json_output')
    pages_color_root = Path(args.pages_color_root)  # Path('/trunk/nvog/print-probability/pages_color')
    #pages_binarized_root = Path('/trunk/nvog/print-probability/pages_binarized')
    #char_binarized_root = Path('/trunk/nvog/print-probability/char_images_binarized')
    #char_color_root = Path('/trunk/nvog/print-probability/char_images3')
    book_char_images_tar = Path(args.book_char_images_tar)
    csv_outfile = Path(args.csv_outfile)  #Path('/trunk/nvog/print-probability/binarized_chars.csv')
    #books = list(json_output_root.glob('*'))
    
    # read in book_char_images_tar file entries
    tarfile_path = book_char_images_tar  # char_color_root/(str(book.name) + '.tar')
    book_char_images_filenames = set()
    print('Reading tar file listing...')
    with tarfile.open(tarfile_path, "r") as tar:
        for tarinfo in tar:
            if tarinfo.isreg():
                book_char_images_filenames.add(str(Path(str(tarinfo.name)).name))
    print('\n'.join([b for b in sorted(book_char_images_filenames)]))
    print('Done.')

    with open(csv_outfile, 'w', newline='') as csvfile:
        fieldnames = ['book', 'char', 'book_char_tar_filepath', 'char_filepath_in_tar', 'char_ocular_logprob', 'char_mean_bin_threshold']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        book = Path(str(args.book_char_images_tar).replace('_uc.tar', '').replace('_lc.tar', ''))
        #import ipdb; ipdb.set_trace()
        #tarfile_path.parent.mkdir(exist_ok=True, parents=True)
        book_pages_json_path = json_output_root/str(book.name)/"pages.json"
        book_chars_json_path = json_output_root/str(book.name)/"chars.json"

        print('Loading json files...')
        with open(book_pages_json_path) as pf, open(book_chars_json_path) as cf:
            pages = json.load(pf)
            chars = json.load(cf)
        print('Done')
            
        print('Extracting char bboxes...')
        char_bboxes_by_page = extract_char_bboxes_by_page_from_json(chars)    
        print('Done.')
        
        print(f"Binarizing {len(pages['pages'])} pages and saving csv rows...")
        for page in pages['pages']:
            #page_img_path = pages_color_root/str(page["filename"]).lstrip('/')
            page_img_path = Path(str(page["filename"]).replace('/pylon5/hm560ip/mpwillia/pics', '/ocean/projects/hum160002p/shared/books').replace('/pylon5/hm4s82p', '/ocean/projects/hum160002p'))
            print('Page:', page_img_path)
            try:
                page_img = Image.open(page_img_path) 
            except FileNotFoundError as e:
                print(page_img_path, 'not found. Skipping...')
                continue

            bin_img, local_thresholds = binarize_img(page_img, window_size=25)
            # save bin image in the pages_binarized dir
            #bin_img_dest = pages_binarized_root/str(page["filename"]).lstrip('/')
            #bin_img_dest.parent.mkdir(exist_ok=True, parents=True)
            #bin_img.save(bin_img_dest, quality=100, subsampling=0)
            # extract chars using the bboxes from chars.json
            # first, find page no. in page filename
            filename = str(page_img_path)
            if '_page' in filename:
                right_bound_idx = filename.rfind('_page')
                pagenum = filename[filename[:right_bound_idx].rfind('-') + 1: right_bound_idx]
            else:
                pagenum = page_img_path.with_suffix('').name.split('-')[-1]
            
            print('Page', pagenum, 'found', len(char_bboxes_by_page[pagenum]), 'char bboxes.')
            # then, crop each char bbox on this page and save it to disk
            for char_bbox, char_filename, char_logprob in char_bboxes_by_page[pagenum]:
                #if char_filename.endswith('uc.tif'):  # NOTE: only use uppercase chars
                if Path(char_filename).name not in book_char_images_filenames:
                    print(f'Skipping character because {Path(char_filename).name} not in tar file list (file name list printed above).')
                    continue
                l, t, r, b = char_bbox
                l = max(0, l)
                t = max(0, t)
                r = max(0, r)
                b = max(0, b)
                
                char_bbox_thresholds = local_thresholds[t:b, l:r]
                if 0 in char_bbox_thresholds.shape:
                    # skip characters with bad bounding boxes
                    print('Skipping bad bbox.')
                    continue
                #char_bbox_bin = np.array(bin_img)[t:b, l:r] 
                #bin_char_crop_img = bin_img.crop(char_bbox)
                #bin_char_crop_img_dest = char_binarized_root/char_filename
                #bin_char_crop_img_dest.parent.mkdir(exist_ok=True, parents=True)
                #bin_char_crop_img.save(str(bin_char_crop_img_dest), quality=100, subsampling=0)

                #color_char_img = page_img.crop(char_bbox)
                #color_char_crop_img_dest = char_color_root/char_filename
                #color_char_crop_img_dest.parent.mkdir(exist_ok=True, parents=True)
                #color_char_img.save(str(color_char_crop_img_dest))
                
                # save color_char_img to tar
                #with io.BytesIO() as output:
                #    color_char_img.save(output, format='TIFF')
                    #contents = output.getvalue()
               #     ti = tarfile.TarInfo(str(color_char_crop_img_dest))
                    #tf.size = len(my_content)
               #     tar.addfile(ti, output)

                #import ipdb; ipdb.set_trace()

                row = {
                        'book': str(book.name),
                        'char': str(char_filename),
                        'book_char_tar_filepath': str(tarfile_path), 
                        'char_filepath_in_tar': str(char_filename),  # str(color_char_crop_img_dest), 
                        'char_ocular_logprob': float(char_logprob), 
                        'char_mean_bin_threshold': float(np.mean(char_bbox_thresholds))
                }
                if str(float(np.mean(char_bbox_thresholds))) == 'nan':
                    print('nan encountered at row:')
                    print(row)
                    continue
                #print('Row', row)
                writer.writerow(row)
            print('Done.')
 
