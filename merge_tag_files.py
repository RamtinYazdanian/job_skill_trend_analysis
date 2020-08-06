import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag_files', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    filenames = args.tag_files.split(',')
    tag_df = [pd.read_csv(filename) for filename in filenames]
    result_df = pd.concat(tag_df)
    result_df = result_df.drop_duplicates()
    result_df.to_csv(os.path.join(args.output_dir, 'so_tags.csv'))

if __name__ == '__main__':
    main()