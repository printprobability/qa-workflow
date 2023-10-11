"""
Creates a directory with copied books for testing autocrop changes.
"""
import os
import shutil
import sys
from pathlib import Path


def setup_autocrop_test_dir(args):
    """ Copies num_books randomly sampled from all collections in books_root to output_dir. """
    # Set random seed
    import random
    random.seed(args.seed)

    # Get all books
    books = []
    if args.add_single_book:
        # get single specified book
        num_books = 1
        books.append(args.add_single_book)
    else:
        # get list of all candidate books from collections
        num_books = args.num_books
        for collection in args.collections:
            collection_path = Path(args.books_root) / collection
            books.extend([str(book) for book in collection_path.glob('*') if book.is_dir() and 'originals_precrop' in os.listdir(book) and len(list((book/'originals_precrop').glob(f'*{args.ext}'))) > args.min_originals])
        
    # Sample num_books
    books = random.sample(books, num_books)
    
    # Copy books to output_dir
    Path(args.output_dir).mkdir(exist_ok=True)
    for book in books:
        print(book)
        # transfer to top-level book directory to prep for autocropping
        dest_dir = Path(args.output_dir)/Path(book).name
        dest_dir.mkdir(parents=True, exist_ok=True)
        pages = sorted((Path(book)/'originals_precrop').glob(f'*{args.ext}'))
        # skip first and last portions of books
        start_page = int(args.skip_start_pages_pct * len(pages))
        end_page = int((1 - args.skip_end_pages_pct) * len(pages))
        for page in pages[start_page:end_page]:
            print(f"Copying {page} to {dest_dir}")
            shutil.copy(page, dest_dir)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='Output directory containing copied books')
    parser.add_argument('--add_single_book',
                        type=str, 
                        help="Optional: Add a single book's path to the directory. This is useful if the test directory"
                             "is already created and we want to add another test book that may be an interesting test case.", 
                        default=None)
    parser.add_argument('--collections', nargs='+', help='Collections to copy books from', default=['restoration', 'shakespeare'])
    parser.add_argument('--num_books', type=int, help='Number of books to copy', default=50)
    parser.add_argument('--seed', type=int, help='Seed for random number generator', default=42)
    parser.add_argument('--books_root', help='Root directory containing books', default='/ocean/projects/hum160002p/shared/books')
    parser.add_argument('--min_originals', help='Minimum amount of images a book must have to be considered as a test book candidate', default=40)
    parser.add_argument('--skip_start_pages_pct', help='Skip pct number of pages at start of book', default=0.15)
    parser.add_argument('--skip_end_pages_pct', help='Skip pct number of pages at end of book', default=0.05)
    parser.add_argument('--ext', help='Page image file extension', default='tif')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    setup_autocrop_test_dir(args)
