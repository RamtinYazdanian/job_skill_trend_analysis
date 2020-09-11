from utilities.common_utils import *
from utilities.pandas_utils import *
from utilities.constants import *
from functools import reduce
from collections import Counter


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


def get_responses_in_rows(initial_dfs, col_type='Skills'):
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
    questions_separate = [questions_separate[i].rename(columns=old_to_new_names[i]).iloc[2:].stack()
                          for i in range(len(questions_separate))]

    questions_separate = [pd.DataFrame(questions_separate[i].stack()).rename(columns={0: QUESTION_NAMES[i]})
                          for i in range(len(questions_separate))]

    df_final = reduce(lambda left, right: pd.merge(left, right, how='outer', left_index=True, right_index=True),
                  questions_separate).reset_index().drop(columns=['level_0']).rename(columns={'level_1': col_type})

    free_text_results = get_free_text_results(dfs, col_type)

    return df_final, free_text_results

def in_depth_answer_to_ground_truth(df, time_periods=TIME_PERIODS):
    df['majority'] = df.apply(lambda x: 1 if Counter(x['Main'])[YES] >=
                                             Counter(x['Main'])[NO] + Counter(x['YesCol'])[BEFORE17] else 0, axis=1)
    for time_period_key in time_periods:
        pass
        # First remove the "Earlier than 2017"s, then for each skill and each time period, compute the number of
        # YesCol responses that include that time period (inclusion criterion: the two periods have to overlap, which,
        # for practical purposes, means that the number in the response - e.g. "Since 2018" - should be strictly
        # smaller than the end year of the time period but also greater than or equal to the starting year of the
        # time period). If this number is equal to, or more than the number of total YesCol responses
        # given for that skill divided by 2 (excluding the "Earlier than 2017" responses), then it's a
        # "1" for the ground truth column for that time period, which essentially means that we treat the
        # overlapping period responses as positives and non-overlapping period responses as negatives.
        # Each time period will have one ground truth column
        # in which each skill has either 0 or 1, and at the end, we create a dictionary where each time period is
        # mapped to a list of the skills that had a 1 in that period's column.

        #modified_yescol_list = df['YesCol'].apply(lambda x: [1 if y. for y in x])

def find_majority_response(df, col_type='Skills', remove_unsure=True):
    assert col_type in ['Skills', 'Firms']
    if remove_unsure:
        df = df.loc[df['Main'] != UNSURE]
    majorities = df.groupby(col_type).agg(lambda x: list(x))


    pass

def assemble_question_responses(initial_dfs, has_firms=True):
    pass
    #main_q =