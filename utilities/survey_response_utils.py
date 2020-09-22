from utilities.common_utils import *
from utilities.pandas_utils import *
from utilities.constants import *
from functools import reduce
from collections import Counter
import krippendorff


def get_skill_or_firm_name(s):
    split_by_questionmark = s.split('? - ')[-1]
    split_by_period = s.split('. - ')[-1]

    if len(split_by_period) < len(split_by_questionmark):
        return split_by_period
    else:
        return split_by_questionmark


def get_free_text_results(df, col_type='Skills'):
    result = df.iloc[2:][[col_type + ', free text']].reset_index().drop('index')
    result[col_type + ', free text'] = result[col_type + ', free text'].apply(lambda x: x.split('\n'))
    return result.explode(col_type + ', free text')


def get_responses_in_rows(initial_dfs, col_type='Skills', filter_nonsense=True):
    """
    Turns the dataframe from the original Qualtrics format into the final desired format, where each row is identified
    by *one* skill/firm and where the columns contain *one person's* responses for that skill/firm. Also returns the
    free text responses in a separate df.
    :param initial_dfs: List of dataframes containing the responses. The responses are the in default Qualtrics
            CSV format, meaning that the first two rows aren't responses, but are rather questions and metadata,
            respectively.
    :param col_type: Skills or Firms. The two types are processed separately but in a similar way.
    :return: Returns two dataframes:
            1) A dataframe with three columns: Main, YesCol, and NoCol; each row is one person's response to
            *one* row of the grid questions.
            2) A dataframe containing the free text responses of users.
    """

    assert col_type in ['Skills', 'Firms']

    all_cols = initial_dfs[0].columns
    cols_to_keep = [colname for colname in all_cols if colname.split(',')[0] == col_type]
    initial_dfs = [df[cols_to_keep] for df in initial_dfs]
    initial_dfs = [initial_dfs[0]] + [initial_dfs[i].iloc[2:] for i in range(1,len(initial_dfs))]
    dfs = pd.concat(initial_dfs, axis=0)

    question_type_cols = [[colname for colname in dfs.columns if 'grid#'+str(i)+'_' in colname] for i in range(1,4)]
    old_to_new_names = [
        {colname: get_skill_or_firm_name(dfs.iloc[0][colname]) for colname in current_names}
        for current_names in question_type_cols
    ]

    questions_separate = [dfs[current_names] for current_names in question_type_cols]
    questions_separate = [questions_separate[i].rename(columns=old_to_new_names[i]).iloc[2:]
                          for i in range(len(questions_separate))]

    questions_separate = [pd.DataFrame(questions_separate[i].stack()).rename(columns={0: QUESTION_NAMES[i]})
                          for i in range(len(questions_separate))]

    df_final = reduce(lambda left, right: pd.merge(left, right, how='outer', left_index=True, right_index=True),
                  questions_separate).reset_index().drop(columns=['level_0']).rename(columns={'level_1': col_type})

    if filter_nonsense:
        # Filters out the responses where the main answer is NaN
        df_final = df_final.loc[~df_final['Main'].isnull()]
        # Corrects responses where, for example, the respondent didn't say No, but still chose an option in the NoCol
        df_final['YesCol'] = df_final.apply(lambda x: x['YesCol'] if x['Main'] == YES else np.nan)
        df_final['NoCol'] = df_final.apply(lambda x: x['NoCol'] if x['Main'] == NO else np.nan)

    free_text_results = get_free_text_results(dfs, col_type)

    return df_final, free_text_results

def does_overlap(time_period_pair):
    starts = [int(x.split('-')[0]) for x in time_period_pair]
    ends = [int(x.split('-')[1]) for x in time_period_pair]
    return starts[0] < ends[1] and starts[1] < ends[0]

def in_depth_answer_to_ground_truth(df, col_type, time_periods=TIME_PERIODS):
    df['majority'] = df.apply(lambda x: 1 if Counter(x['Main'])[YES] >=
                                             Counter(x['Main'])[NO] + Counter(x['YesCol'])[BEFORE17] else 0, axis=1)
    modified_yescol_list = df['YesCol'].apply(lambda x: [y for y in remove_list_nans(x) if y != BEFORE17]).apply(
                        lambda x: [y.split(' ')[1]+'-2020' for y in x]
    )
    df['modified_yescol'] = modified_yescol_list

    result_dict = dict()
    for time_period_key in time_periods:
        # First remove the "Earlier than 2017"s, then for each skill and each time period, compute the number of
        # YesCol responses that include that time period (inclusion criterion: the two periods have to overlap, which,
        # for practical purposes, means that the number in the response - e.g. "Since 2018" - should be strictly
        # smaller than the end year of the time period but also greater than or equal to the starting year of the
        # time period). If this number is equal to, or more than the number of total YesCol responses
        # given for that skill divided by 2 (excluding the "Earlier than 2017" responses), then it's a
        # True for the ground truth column for that time period, which essentially means that we treat the
        # overlapping period responses as positives and non-overlapping period responses as negatives.
        # Each time period will have one ground truth column
        # in which each skill has either False or True, and at the end, we create a dictionary where each time period is
        # mapped to a list of the skills that had a True in that period's column.

        new_period_col = df.apply(lambda x: (len([y for y in x['modified_yescol'] if does_overlap([time_period_key, y])]),
                                             len(x['modified_yescol'])) if x['majority'] == 1 else (-1,-1), axis=1)
        new_period_col = new_period_col.apply(lambda x: x[0] >= x[1]/2 if x[1] > 0 else False)
        df[time_period_key] = new_period_col
        result_dict[time_period_key] = df.loc[df[time_period_key] == True, col_type].values.tolist()

    return df, result_dict

def find_majority_response_ground_truth(df, time_periods=TIME_PERIODS, col_type='Skills'):
    """
    Computes the majority response for each skill/firm, removing Unsure responses and treating "Before 2017" as a No.
    Then, creates a ground truth dictionary wherein each time period is mapped to the skills/firms that qualified as
    emerging/trend-anticipating in that time period, based on their majority (Yes) and YesCol (time period overlap for
    more than half) responses.
    :param df: The starting df, which is the output of get_responses_in_rows
    :param time_periods: A dictionary where the keys are the string representations of time periods, like '2018-2020'.
    :param col_type: Skills or Firms
    :return: The majority dataframe with time period columns, plus a ground truth dictionary mapping each time period
    key to the list of skills/firms that qualified for ground truth in that period.
    """
    assert col_type in ['Skills', 'Firms']
    df = df.loc[df['Main'] != UNSURE]
    majorities = df.groupby(col_type).agg(lambda x: list(x)).reset_index()
    majorities, ground_truth_for_periods = in_depth_answer_to_ground_truth(majorities, col_type, time_periods)

    return majorities, ground_truth_for_periods

def assemble_question_responses(initial_dfs, has_firms=True):
    skills = get_responses_in_rows(initial_dfs, col_type='Skills')
    if has_firms:
        firms = get_responses_in_rows(initial_dfs, col_type='Firms')
    else:
        firms = None
    return skills, firms

def get_response_proportions(df, col_type='Skills', col_to_analyse='Main'):
    df = df[[col_type, col_to_analyse]].groupby(col_type).agg(list).reset_index()
    df[col_to_analyse] = df[col_to_analyse].apply(remove_list_nans)
    df[col_to_analyse+'_response_count'] = df[col_to_analyse].apply(len)
    df[col_to_analyse] = df[col_to_analyse].apply(lambda x: Counter(x))
    df[col_to_analyse+'_proportions'] = df.apply(lambda x: [(i, y/x[col_to_analyse+'_response_count']) for i, y in
                                                            x[col_to_analyse].items()], axis=1)
    return df

def get_value_counts_for_agreement(df, col_to_analyse, remove_unsure=False):
    """
    :param df: Dataframe from get_response_proportions
    :param col_to_analyse: Name of the column to use, default 'Main'.
    :param remove_unsure: Whether or not to remove Unsure from the responses
    :return: Returns a (n_skills * 2 or 3) array that counts each type of response (Yes, No, optionally Unsure).
    """
    df[YES] = df[col_to_analyse].apply(lambda x: x[YES])
    df[NO] = df[col_to_analyse].apply(lambda x: x[NO])
    cols_to_keep = [YES, NO]
    if not remove_unsure:
        df[UNSURE] = df[col_to_analyse].apply(lambda x: x[UNSURE])
        cols_to_keep.extend([UNSURE])
    df = df[cols_to_keep]
    return df.values

def get_interrater_agreement(df, col_type='Skill', remove_unsure=False):
    df = get_response_proportions(df, col_type, 'Main')
    value_counts = get_value_counts_for_agreement(df, 'Main', remove_unsure)
    return krippendorff.alpha(value_counts=value_counts, level_of_measurement='nominal')