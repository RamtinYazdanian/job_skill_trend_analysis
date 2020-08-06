import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from utilities.pandas_utils import get_period_of_time

def divide_into_periods(df, cols, start_date=None, end_date=None, result_col_name='Count'):
    df = get_period_of_time(df, start_date, end_date).copy()
    df['Date'] = df['Date'].apply(lambda x: np.datetime64(str(x.year) + '-' + '{:0>2}'.format(x.month) + '-01'))
    df[result_col_name] = 1
    df = df[cols+['Date', result_col_name]].groupby(cols+['Date']).sum().\
                            reset_index()
    return df


def delete_low_freq_skills(df, min_freq):
    skill_counts = df.groupby('Skill').count().reset_index()
    low_count_skills = skill_counts.loc[skill_counts['Job Postings'] < min_freq].Skill.values
    return df.loc[df.Skill.apply(lambda x: x not in low_count_skills)]


def fill_in_the_blank_dates(df, method='zero', has_company=True):
    base_cols = ['Date']
    columns_list = ['Job Postings', 'Job Postings Raw', 'Total']
    filling_df = df[base_cols + columns_list].groupby(base_cols)
    if method == 'min':
        filling_df = filling_df.min()
    elif method == 'mean':
        filling_df = filling_df.mean()
    elif method == 'median':
        filling_df = filling_df.median()
    elif method == 'zero':
        filling_df = filling_df.min()
        for col in columns_list:
            filling_df[col] = 0
    else:
        return df
    filling_df = filling_df.reset_index().assign(key_col=1)
    skills_only_df = df[['Skill']].drop_duplicates().assign(key_col=1)
    print('Number of skills: ' + str(skills_only_df.shape[0]))

    filling_df = pd.merge(filling_df, skills_only_df, on='key_col')
    if has_company:
        companies = df[['Company']].copy()
        print('Number of companies: ' + str(companies.shape[0]))
        companies['key_col'] = 1
        filling_df = pd.merge(filling_df, companies, on='key_col')
    filling_df = filling_df.drop(columns=['key_col'])

    result_df = pd.merge(df, filling_df, how='right', on=['Skill'] + base_cols,
                         suffixes=('_actual', '_filled'))

    for col in columns_list:
        result_df[col + '_filled'] = result_df.apply(lambda x: x[col + '_filled'] if pd.isnull(x[col + '_actual']) else
        x[col + '_actual'], axis=1)
    result_df = result_df.drop(columns=[x + '_actual' for x in columns_list]).rename(
        columns={x + '_filled': x for x in columns_list})
    return result_df


def group_time_steps_together(df, steps_to_group=3, has_company=True):
    if steps_to_group == 1:
        return df
    if has_company:
        base_cols = ['Date', 'Company']
    else:
        base_cols = ['Date']
    dates_df = df[base_cols].drop_duplicates().sort_values(base_cols).reset_index().drop(columns=['index'])
    dates_df['group'] = pd.Series(list(range(dates_df.shape[0])))
    dates_df['group'] = dates_df['group'].apply(lambda x: 1 + (x // steps_to_group))
    print(dates_df)
    min_dates = dates_df.groupby('group').min().reset_index()
    result_df = pd.merge(df, dates_df, on=base_cols).drop(columns=['Date']).groupby(['group', 'Skill']).mean(). \
        reset_index()
    result_df = pd.merge(result_df, min_dates, on='group').drop(columns=['group'])
    return result_df

def linreg_jobpostings(df, y_col='Job Postings', normaliser=None, smooth='exp', log=True):
    if smooth is not None:
        if smooth == 'movingavg':
            y = df[y_col].rolling(3).mean().values
            y = y[2:]
        elif smooth == 'exp':
            y = df[[y_col]].ewm(alpha=0.8, adjust=False).mean().values
        else:
            y = df[[y_col]].values
    else:
        y = df[[y_col]].values
    if normaliser is None:
        y = PolynomialFeatures(degree=2, include_bias=False).fit_transform(y)
    else:
        if log:
            y = PolynomialFeatures(degree=2, include_bias=False).fit_transform(y -
                                                                           normaliser[['Total']].values)
        else:
            y = PolynomialFeatures(degree=2, include_bias=False).fit_transform(y /
                                                                               normaliser[['Total']].values)
    X = df[['Date']].values
    X = (X - X.min()).astype('timedelta64[D]') / np.timedelta64(1, 'D') / 30
    if len(y) < len(X):
        X = X[-len(y):]
    result_model = LinearRegression()
    result_model.fit(X, y) # Weighting the first point makes no conceptual sense because the 1st point isn't special.
    spike_value = y.max() / y.mean()
    return result_model.coef_[0][0], result_model.coef_[1][0], spike_value, result_model.intercept_[0]

def get_trend_slope_intercept(group_col_and_trends):
    group_col_and_trends['Slope'] = group_col_and_trends[0].apply(lambda x: x[0] if not
                                                                               pd.isna(x) else np.nan)
    group_col_and_trends['Intercept'] = group_col_and_trends[0].apply(lambda x: x[3] if not
                                                                               pd.isna(x) else np.nan)
    group_col_and_trends['Acceleration'] = group_col_and_trends[0].apply(lambda x: x[1] if not
                                                                               pd.isna(x) else np.nan)
    group_col_and_trends['Spikiness'] = group_col_and_trends[0].apply(lambda x: x[2] if not
                                                                               pd.isna(x) else np.nan)
    group_col_and_trends = group_col_and_trends.drop(columns=0)
    return group_col_and_trends

def compute_total_log_mean(df):
    """
    For each date, computes the average, among all companies, of the log of their total number of ads.
    """
    return df[['Date', 'Company', 'Total']].drop_duplicates().drop(columns=['Company'])\
                    .groupby('Date').apply(lambda x: np.mean(x['Total'].apply(np.log))).reset_index().rename(
                                                            columns={0: 'Total'})

def compute_total_values(df):
    return df[['Date', 'Company', 'Total']].drop_duplicates().groupby('Date').\
                                                                sum().reset_index()

def logsum_trend_slope_wrapper(df, starting_date, end_date, total_log, min_freq=1, grouping=1,
                               nafill='zero', nologtest=False):
    """
    Computes the dataframe containing the log sum trends (slope, intercept, acceleration, etc.) based on
    a skills dataframe. The starting dataframe needs to have the columns 'Date', 'Skill', and
    'Job Postings Raw', and needs to be company-level. The log of the company-level values is taken,
    and then they are summed up, grouped by skill and date.
    """
    print('Start: ' + str(starting_date))
    print('End: ' + str(end_date))
    df = get_period_of_time(df, starting_date, end_date).copy()
    skills_raw_sums = df[['Skill', 'Job Postings Raw']].groupby('Skill').sum()
    if not nologtest:
        df['Job Postings'] = df['Job Postings Raw'].apply(lambda x: np.log(1+x))

    df = df.groupby(['Date', 'Skill']).sum().reset_index()

    df_with_trends_pooled = pd.DataFrame(
        fill_in_the_blank_dates(
            group_time_steps_together(
                delete_low_freq_skills(df, min_freq),
                            steps_to_group=grouping, has_company=False), method=nafill, has_company=False).
                                 groupby('Skill').apply(lambda x:
                                    linreg_jobpostings(x, normaliser=
                                           get_period_of_time(total_log, starting_date,
                                                              end_date))))
    df_with_trends_pooled = get_trend_slope_intercept(df_with_trends_pooled)
    print(df_with_trends_pooled.sort_values('Slope', ascending=False).describe())
    return df_with_trends_pooled.join(skills_raw_sums)

def threshold_logsum_trends_simple(df_with_trends, total, col='Slope', col_percentile_thresh=.7, col_std_thresh=0,
                                   only_positives = True,
                                   pop_lower=0.001, pop_upper=0.01):
    pop_lower = pop_lower*total
    pop_upper = pop_upper*total

    if only_positives:
        df_with_trends = df_with_trends.loc[df_with_trends[col] > 0]

    if col_percentile_thresh is None:
        col_thresh = df_with_trends[col].mean() + col_std_thresh * df_with_trends[col].std()
    else:
        col_thresh = df_with_trends[col].quantile(col_percentile_thresh)

    return df_with_trends.loc[(df_with_trends[col] >= col_thresh) &
                              (df_with_trends['Job Postings Raw'] >= pop_lower) &
                              (df_with_trends['Job Postings Raw'] < pop_upper)].reset_index()

def merge_skill_with_score(df, skills, col, sort_type):
    if sort_type == 'score':
        return sorted([(skill, df.loc[df.Skill == skill, col].values[0]) for skill in skills], key=lambda x: -x[1])
    else:
        return sorted([(skill, df.loc[df.Skill == skill, col].values[0]) for skill in skills], key=lambda x: x[0])

def compare_emerging_skill_sets(emerging_skills, dates, sort_type='score'):
    for i in range(len(emerging_skills)):
        for j in range(i+1,len(emerging_skills)):
            print('\nComparing ' + ' to '.join([str(dates[i][k]) for k in range(2)]) + ' with ' +
                                            ' to '.join([str(dates[j][k]) for k in range(2)]))
            skills_shared = set(emerging_skills[i].Skill.values).intersection(set(emerging_skills[j].Skill.values))
            skills_exclusive_i = \
                        set(emerging_skills[i].Skill.values).difference(set(emerging_skills[j].Skill.values))
            skills_exclusive_j = \
                        set(emerging_skills[j].Skill.values).difference(set(emerging_skills[i].Skill.values))
            skills_shared = merge_skill_with_score(emerging_skills[i], skills_shared, 'Slope', sort_type)
            skills_exclusive_i = merge_skill_with_score(emerging_skills[i], skills_exclusive_i, 'Slope', sort_type)
            skills_exclusive_j = merge_skill_with_score(emerging_skills[j], skills_exclusive_j, 'Slope', sort_type)
            print('\n# of skills shared and exclusive to each date (in order)')
            print(len(skills_shared), len(skills_exclusive_i), len(skills_exclusive_j))
            print('\nThe skills themselves:')
            print('\nShared:' + '\n')
            print([x[0] for x in skills_shared])
            print('\nExclusive to '+ ' to '.join([str(dates[i][k]) for k in range(2)]) + '\n')
            print([x[0] for x in skills_exclusive_i])
            print('\nExclusive to '+ ' to '.join([str(dates[j][k]) for k in range(2)]) + '\n')
            print([x[0] for x in skills_exclusive_j])

def compute_prec_recall(predicted_set, reference_set):
    predicted_set = predicted_set.Skill.values
    reference_set = reference_set.Skill.values
    accurately_predicted = set(reference_set).intersection(set(predicted_set))
    print(accurately_predicted)
    print(len(accurately_predicted), len(predicted_set), len(reference_set))
    if len(predicted_set) > 0:
        prec = len(accurately_predicted) / len(predicted_set)
        recall = len(accurately_predicted) / len(reference_set)
        f1 = 2*prec*recall / (prec+recall)
    else:
        prec = 0
        recall = 0
        f1 = 0
    return prec, recall, f1