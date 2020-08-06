import pandas as pd
from utilities.common_utils import *
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matched', type=str, required=True)
    parser.add_argument('--unmatched', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    matched_df = pd.read_csv(args.matched)
    unmatched_df = pd.read_csv(args.unmatched)

    it_skills = pd.concat([matched_df, unmatched_df.loc[unmatched_df['is_it'] == 1, ['Skill']]])
    non_it_skills = unmatched_df.loc[unmatched_df['is_it'] == 2]

    it_skills.to_csv(os.path.join(args.output_dir, 'it_skills.csv'), index=False)
    non_it_skills.to_csv(os.path.join(args.output_dir, 'other_skills.csv'), index=False)


if __name__ == '__main__':
    main()