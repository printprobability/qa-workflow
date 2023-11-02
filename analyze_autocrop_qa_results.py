"""
Make plots and show images from each book for each method at the 25th, 50th, and 75th percentiles of the distribution of the metric of interest
"""

import imgcat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def make_stripplot(args):
    import seaborn as sns
    sns.set(style='white')
    metric = args.metric
    options = args.autocrop_options
    df = pd.read_csv(args.result_csv)
    fig, axes = plt.subplots(figsize=(10, 30))
    axes.set_title('Comparing autocrop settings')
    df_options = df[df['autocrop_type'].isin(options)].sort_values(by=['book_name'], ascending=True)
    df_options[f'{metric}_median'] = df_options.groupby(['book_name'])[[metric]].transform('median')
    grouped = df_options.groupby(['book_name', 'autocrop_type'])
    df_options.sort_values(by=[f'{metric}_median'], inplace=True)
    sns.stripplot(data=df_options, hue='autocrop_type', y='book_name', x=metric, ax=axes, dodge=True)
    axes.grid(True)
    axes.set_ylabel('Book string')
    axes.set_xlabel(metric.replace('_', ' '))
    #plt.show()
    fname = f'autocrop_qa_{metric}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    imgcat.imgcat(Image.open(fname))


def show_sample_images(args):
    metric = args.metric
    options = args.autocrop_options
    df = pd.read_csv(args.result_csv)
    df_options = df[df['autocrop_type'].isin(options)].sort_values(by=['book_name'], ascending=True)
    df_options[f'{metric}_median'] = df_options.groupby(['book_name'])[[metric]].transform('median')
    df_options.sort_values(by=[f'{metric}_median'], inplace=True)
    grouped = df_options.groupby(['book_name', 'autocrop_type'], sort=False)
    # grouped = df_options.groupby(['book_name', 'autocrop_type'])[metric].quantile([0.25, 0.5, 0.75], interpolation='nearest')
    # print(grouped)

    quantiles = args.quantiles
    print(f'Printing images for each book for each method at the {quantiles} percentiles of the distribution of the metric {metric}')
    for name, group in grouped:
        book_name = name[0]
        autocrop_type = name[1]
        this_df = df_options[(df_options.book_name == book_name) & (df_options.autocrop_type == autocrop_type)]
        qs = this_df[metric].quantile(quantiles, interpolation='nearest')
        print(book_name, autocrop_type, qs.tolist())
        for q in qs:
            # show image with this nearest quantile metric value
            # print(this_df[this_df[metric] == q].image_name)
            # example img:
            #/ocean/projects/hum160002p/shared/books/test_qa/test_autocrop/anon_R11260_wellcome_4_generalhistoryair1692/results/non_threshold_by_inside
            fname = this_df[this_df[metric] == q].image_name.tolist()[0]
            fpath = str(Path(args.test_autocrop_dir)/book_name/'results'/autocrop_type/fname)
            if args.debug:
                print(fpath)
            im = Image.open(fpath)
            w, h = im.size
            imgcat.imgcat(im.resize((w // 3, h // 3)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze autocrop test results.')
    parser.add_argument('--result_csv', type=str, default='/ocean/projects/hum160002p/shared/books/code/qa_workflow/logs/all_results_merged_6bfe61cf-1f60-4b08-bd31-25bf09e301a2.csv', help='Path to merged result csv')
    parser.add_argument('--metric', type=str, default='percent_area_diff_from_original', choices=['percent_area_diff_from_original', 'min_pct_dimension_difference'], help='Metric to use for analysis')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.25, 0.5, 0.75])
    parser.add_argument('--test_autocrop_dir', default='/ocean/projects/hum160002p/shared/books/test_qa/test_autocrop')
    parser.add_argument('--autocrop_options', type=str, nargs='+', default=['non_threshold_by_inside', 'threshold_by_inside'], help='autocrop options to compare')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    make_stripplot(args)
    show_sample_images(args)
