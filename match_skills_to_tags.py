import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tags', type=str, required=True)
    parser.add_argument('--skills', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    skills_df = pd.read_csv(args.skills)
    tags_df = pd.read_csv(args.tags)

    skills_df['TagName'] = skills_df['Skill'].apply(lambda x: x.replace(' ', '-'))
    matched_skills = pd.merge(skills_df, tags_df, on='TagName', how='inner')[['Skill']]
    matched_set = set(matched_skills.Skill.values)
    unmatched_skills = skills_df.loc[skills_df.Skill.apply(lambda x: x not in matched_set), ['Skill']]

    matched_skills.to_csv(os.path.join(args.output_dir, 'matched_skills.csv'))
    unmatched_skills.to_csv(os.path.join(args.output_dir, 'unmatched_skills.csv'))


if __name__ == '__main__':
    main()