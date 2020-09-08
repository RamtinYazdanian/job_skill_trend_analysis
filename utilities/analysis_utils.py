import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from utilities.pandas_utils import get_period_of_time
from utilities.constants import *

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

def smooth_and_normalise_timeseries(df, log, normaliser, smooth, y_col, date_to_step=True, return_df=False):
    """
    Performs smoothing and normalisation on the output values and converts the dates into input values.
    :param df:
    :param log:
    :param normaliser:
    :param smooth:
    :param y_col:
    :return:
    """

    # Normalising and smoothing y
    y = df[[y_col]].values
    if smooth is not None:
        if smooth == 'movingavg':
            y = df[y_col].rolling(SMOOTH_N).mean().values
            y = y[SMOOTH_N - 1:]
        elif smooth == 'exp':
            y = df[[y_col]].ewm(alpha=SMOOTH_ALPHA, adjust=False).mean().values
    if normaliser is not None:
        if smooth is not None:
            if smooth == 'movingavg':
                normaliser = normaliser[['Total']].rolling(SMOOTH_N).mean().values
                normaliser = normaliser[SMOOTH_N - 1:]
                normaliser = np.reshape(np.array(normaliser), newshape=y.shape)
            elif smooth == 'exp':
                normaliser = normaliser[['Total']].ewm(alpha=SMOOTH_ALPHA, adjust=False).mean().values
                np.reshape(normaliser[['Total']].values, newshape=y.shape)
        if log:
            y = y - normaliser
        else:
            y = y / normaliser
    y = y.flatten()

    # Turning X into steps rather than dates
    X = df[['Date']].values
    if date_to_step:
        X = (X - X.min()).astype('timedelta64[D]') / np.timedelta64(1, 'D') / 30

    # Making sure X and y have the same length (moving avg smoothing will shorten y)
    if len(y) < len(X):
        X = X[-len(y):]

    # Default is to return X and y as arrays, not as a dataframe.
    if not return_df:
        return X, y
    else:
        return pd.DataFrame({'Date': X, y_col: y})

def linreg_jobpostings(df, y_col='Job Postings', normaliser=None, smooth='exp', log=True, degree=2):
    X, y = smooth_and_normalise_timeseries(df, log, normaliser, smooth, y_col)
    X = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)
    result_model = LinearRegression()
    result_model.fit(X, y) # Weighting the first point makes no conceptual sense because the 1st point isn't special.
    spike_value = y.max() / y.mean()
    if degree == 1:
        if isinstance(result_model.intercept_, float):
            return np.array([result_model.coef_[0], 0,
                             spike_value, result_model.intercept_])
        else:
            return np.array([result_model.coef_[0][0], 0,
                             spike_value, result_model.intercept_[0]])
    else:
        if isinstance(result_model.intercept_, float):
            return np.array([result_model.coef_[0], result_model.coef_[1],
                             spike_value, result_model.intercept_])
        else:
            return np.array([result_model.coef_[0][0], result_model.coef_[1][0],
                             spike_value, result_model.intercept_[0]])


def extract_timeseries_features(df, y_col='Job Postings', extraction_method='linreg',
                                normaliser=None, smooth='exp', params=None):
    """
    Extracts features for one skill's time series and returns them as a vector.

    :param df: The groupby dataframe that has the time series for one skill.
    :param y_col: The output column (the input is 'Date' by default)
    :param extraction_method: The type of feature extraction. Supported modes are 'linreg' and 'full'.
    :param normaliser: The dataframe used to normalise the output.
    :param smooth: Smoothing method (None, 'exp', or 'movingavg')
    :param params: Parameters specific to the feature extraction method. These are as follows:
            For all methods:
                'log': Whether the normalisation is logarithmic (subtraction) or not (division)
                'pop_type': Whether the type of popularity used (as the output variable) was log, bin, or raw
            For lingreg:
                'degree': The degree of the polynomial fitted.

            Method-specific parameters should already be provided in the dict given to the wrapper function.
    :return: Returns a feature vector generated for that skill.
    """
    if extraction_method == 'linreg':
        return linreg_jobpostings(df, y_col=y_col, normaliser=normaliser, smooth=smooth,
                                  log=params['log'], degree=params['degree'])
    elif extraction_method == 'full':
        pass


def get_trend_slope_intercept(group_col_and_trends, feature_names):
    for i in range(len(feature_names)):
        feature_name = feature_names[i]
        group_col_and_trends[feature_name] = group_col_and_trends['Features'].apply(lambda x: x[i] if not
                                                                                   pd.isna(x) else np.nan)
    return group_col_and_trends

def compute_hybrid_score(df, col, weights):
    return df[col].apply(lambda x: np.dot(x, weights))


def get_skill_pop_time_series(df, pop_type, params=None):
    assert pop_type in ['log', 'bin', 'raw']
    if params is not None:
        params['type'] = pop_type
    # Using logpop, binpop, or rawpop.
    if pop_type == 'log':
        # log popularity: The per-date company-level value for each skill is log(1+nAds_skill_t)
        df['Job Postings'] = df['Job Postings Raw'].apply(lambda x: np.log(1 + x))
        if params is not None:
            params['log'] = True
    elif pop_type == 'bin':
        # bin popularity: The per-date company-level value for each skill is 1
        df['Job Postings'] = 1
        if params is not None:
            params['log'] = False
    elif pop_type == 'raw':
        # raw popularity: The per-date company-level value for each skill is te raw value, nAds_skill_t
        df['Job Postings'] = df['Job Postings Raw']
        if params is not None:
            params['log'] = False
    df = df.groupby(['Date', 'Skill']).sum().reset_index()
    return df


def skill_trend_features_wrapper(df, starting_date, end_date, total_values, min_freq=1, feature_type='linreg',
                                 nafill='zero', pop_type='log', smoothing='movingavg', params=None, weights=None):
    """

    :param df: The dataframe. It dataframe needs to have the columns 'Date', 'Skill', and
            'Job Postings Raw', and needs to be company-level.
    :param starting_date: The starting date for the period for which features are going to be computed.
    :param end_date: The end date for the period.
    :param total_values: The total values that will be used for normalisation. If pop_type is 'log', this has to be
            a sum of logs, and if it's 'bin', it has to be a sum of binarised values.
    :param min_freq: Minimum frequency for skills. Skills below it are deleted. Default 1.
    :param grouping: How many time steps to group together. Default 1.
    :param feature_type: The type of feature extraction used. Options are 'linreg' and 'full'.
    :param nafill: How to fill null values. Use 'zero'.
    :param pop_type: Type of popularity to use. 'log', 'bin', or 'raw'.
    :param smoothing: Type of smoothing used. None, 'exp', or 'movingavg'.
    :param params: The parameters for the feature extraction method used.
    :param weights: The weights used for the features in order to compute the HybridScore. The score is equal to the
            dot product of the weights vector and the feature vector.
    :return: A dataframe where every skill has its extracted features and potentially the HybridScore.
    """

    if params is None:
        params = dict()

    print('Start: ' + str(starting_date))
    print('End: ' + str(end_date))
    df = get_period_of_time(df, starting_date, end_date).copy()
    skills_raw_sums = df[['Skill', 'Job Postings Raw']].groupby('Skill').sum()

    df = get_skill_pop_time_series(df, params, pop_type)

    df_with_trends_pooled = pd.DataFrame(
        fill_in_the_blank_dates(
                delete_low_freq_skills(df, min_freq), method=nafill, has_company=False).
                                 groupby('Skill').apply(lambda x:
                                    extract_timeseries_features(x, extraction_method=feature_type, normaliser=
                                           get_period_of_time(total_values, starting_date,
                                                              end_date), smooth=smoothing, params=params)))

    df_with_trends_pooled = df_with_trends_pooled.rename(columns={0: 'Features'})
    if feature_type == 'linreg':
        df_with_trends_pooled = get_trend_slope_intercept(df_with_trends_pooled, feature_names=LINREG_FEATURES)


    if weights is not None:
        df_with_trends_pooled['HybridScore'] = compute_hybrid_score(df_with_trends_pooled, 'Features', weights)

    print(df_with_trends_pooled.describe())
    return df_with_trends_pooled.join(skills_raw_sums)


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

def threshold_logsum_trends_simple(df_with_trends, col='HybridScore', pop_col='Job Postings Raw',
                                   col_percentile_thresh=.7, col_std_thresh=0,
                                   only_positives = False,
                                   pop_lower_percentile=0.5, pop_upper_percentile=0.9,
                                   pop_lower=0.001, pop_upper=0.01, total=None):
    """
    Takes the scored skills, thresholds them based on their score and popularity, and returns the "emerging" skills.

    :param df_with_trends: Dataframe that has one row per skill, with that skill's score and popularity
            during the time period in question.
    :param col: The name of the score column.
    :param pop_col: The name of the popularity column.
    :param col_percentile_thresh: Percentile threshold on score col for inclusion in emerging skills. Mutually
            exclusive with col_std_thresh.
    :param col_std_thresh: If col_percentile_thresh is None, then the threshold for emergingness
            is mean + col_std_thresh*std.
    :param only_positives: Whether to discard the negative part of the distribution. False by default.
    :param pop_lower_percentile: Upper bound percentile for Job Postings Raw.
    :param pop_upper_percentile: Lower bound percentile for Job Postings Raw.
    :param pop_lower: Only if both the percentile thresholds are None, this is a percentage of total lower bound.
    :param pop_upper: Only if both the percentile thresholds are None, this is a percentage of total upper bound.
    :param total: Total number of ads in the time period, only applicable for pop_lower and pop_upper.

    :return: The rows of the original Dataframe that correspond to emerging skills per the criteria provided.
    """
    if pop_lower_percentile is None and pop_upper_percentile is None:
        pop_lower = pop_lower*total
        pop_upper = pop_upper*total
    else:
        if pop_lower_percentile is None:
            pop_lower_percentile = 0
        if pop_upper_percentile is None:
            pop_upper_percentile = 1
        pop_lower = df_with_trends[pop_col].quantile(pop_lower_percentile)
        pop_upper = df_with_trends[pop_col].quantile(pop_upper_percentile)

    if only_positives:
        df_with_trends = df_with_trends.loc[df_with_trends[col] > 0]

    if col_percentile_thresh is None:
        col_thresh = df_with_trends[col].mean() + col_std_thresh * df_with_trends[col].std()
    else:
        col_thresh = df_with_trends[col].quantile(col_percentile_thresh)

    return df_with_trends.loc[(df_with_trends[col] >= col_thresh) &
                              (df_with_trends[pop_col] >= pop_lower) &
                              (df_with_trends[pop_col] < pop_upper)].reset_index()

def merge_skill_with_score(df, skills, col, sort_type):
    if sort_type == 'score':
        return sorted([(skill, df.loc[df.Skill == skill, col].values[0]) for skill in skills], key=lambda x: -x[1])
    else:
        return sorted([(skill, df.loc[df.Skill == skill, col].values[0]) for skill in skills], key=lambda x: x[0])

def get_set_intersection_and_diff(set_1, set_2):
    shared_values = set(set_1).intersection(set(set_2))
    exclusive_1 = \
        set(set_1).difference(set(set_2))
    exclusive_2 = \
        set(set_2).difference(set(set_1))
    return exclusive_1, exclusive_2, shared_values


def compare_emerging_skill_sets(emerging_skills, dates, sort_type='score', sort_col='Slope'):
    for i in range(len(emerging_skills)):
        for j in range(i+1,len(emerging_skills)):
            print('\nComparing ' + ' to '.join([str(dates[i][k]) for k in range(2)]) + ' with ' +
                                            ' to '.join([str(dates[j][k]) for k in range(2)]))
            skills_i = emerging_skills[i].Skill.values
            skills_j = emerging_skills[j].Skill.values
            skills_exclusive_i, skills_exclusive_j, skills_shared = get_set_intersection_and_diff(skills_i, skills_j)
            skills_shared = merge_skill_with_score(emerging_skills[i], skills_shared, sort_col, sort_type)
            skills_exclusive_i = merge_skill_with_score(emerging_skills[i], skills_exclusive_i, sort_col, sort_type)
            skills_exclusive_j = merge_skill_with_score(emerging_skills[j], skills_exclusive_j, sort_col, sort_type)
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
    predicted_set = predicted_set.drop_duplicates(subset=['Skill'])
    predicted_set = predicted_set.Skill.values
    reference_set = reference_set.Skill.values
    accurately_predicted = set(reference_set).intersection(set(predicted_set))
    print('\nCorrectly predicted skills:')
    print(accurately_predicted)
    print('\nNumber of correctly predicted skills, all predicted positives, and true positives:')
    print(len(accurately_predicted), len(predicted_set), len(reference_set))
    if len(predicted_set) > 0:
        prec = len(accurately_predicted) / len(predicted_set)
        recall = len(accurately_predicted) / len(reference_set)
        if prec != 0 and recall != 0:
            f1 = 2*prec*recall / (prec+recall)
        else:
            f1 = 0
    else:
        prec = 0
        recall = 0
        f1 = 0
    return prec, recall, f1


def get_responsible_companies(df, skill, time_periods, time_index):
    filtered_df = get_period_of_time(df, time_periods[time_index][0], time_periods[time_index][1])
    filtered_df = filtered_df.loc[(filtered_df.Skill == skill)]
    return filtered_df[['Company', 'Skill', 'Job Postings Raw']].\
                groupby(['Company', 'Skill']).sum().sort_values('Job Postings Raw', ascending=False)

def get_skills_by_threshold(skill_trends, total_values,
                            score_col, score_thresholds, pop_lower_bounds, pop_upper_bounds, sample=None):
    # This function requires each list of thresholds to be progressively *less* restrictive.
    df_results = list()
    set_results = list()

    counter = 0
    for score_thresh in score_thresholds:
        for pop_lower in pop_lower_bounds:
            for pop_upper in pop_upper_bounds:
                    counter += 1
                    new_df = threshold_logsum_trends_simple(skill_trends, total=total_values, col=score_col,
                                   only_positives=True,
                                   col_percentile_thresh=score_thresh,
                                   pop_lower_percentile=pop_lower, pop_upper_percentile=pop_upper)
                    new_set = set(new_df.Skill.values)
                    for existing_set in set_results:
                        new_set = new_set.difference(existing_set)

                    new_df = new_df.loc[new_df.Skill.apply(lambda x: x in new_set)]
                    new_df['Params'] = [tuple([score_thresh, pop_lower, pop_upper])]*new_df.shape[0]
                    new_df['Set#'] = counter

                    if sample is not None and new_df.shape[0] > 0:
                        if sample < 1 and sample > 0:
                            new_df = new_df.sample(frac=sample)
                        elif sample >= 1:
                            new_df = new_df.sample(n=min([sample, new_df.shape[0]]))
                        new_set = set(new_df.Skill.values.tolist())

                    df_results.append(new_df)
                    set_results.append(new_set)

    return pd.concat(df_results).sort_values('Set#').reset_index().drop(columns=['index']), set_results

def get_set_of_companies(df, companies):
    result = df.loc[df.Company.apply(lambda x: x in companies)].copy()
    return result

def hits_on_companies(skills_df, seed_skills, binarise=True, max_iter=20, tol=0.0001, weights=None):
    # This needs to be run on a specific time slice of the skills df (e.g. 2018-2019).
    if not isinstance(seed_skills, set) and not isinstance(seed_skills, list):
        # then it's a dataframe
        seed_skills = set(seed_skills.Skill.values.tolist())
    skills_df = skills_df.loc[skills_df.Skill.apply(lambda x: x in seed_skills), ['Skill', 'Company']]
    if binarise:
        skills_df = skills_df.drop_duplicates()
    companies_only_df = skills_df[['Company']].drop_duplicates()
    companies_only_df['Score'] = 1
    skills_only_df = skills_df[['Skill']].drop_duplicates()
    skills_only_df['Score'] = 1

    finished = False
    iter_count = 0
    skill_total = 0
    company_total = 0

    while not finished:
        iter_count += 1

        # Authority update
        skills_only_new = pd.merge(skills_df, companies_only_df, on='Company').groupby('Skill').sum().reset_index()
        skill_total_new = np.sqrt(skills_only_new['Score'].apply(lambda x: x ** 2).sum())
        skills_only_new['Score'] = skills_only_new['Score'] / skill_total_new

        # Hub update
        companies_only_new = pd.merge(skills_df, skills_only_df, on='Skill').groupby('Company').sum().reset_index()
        company_total_new = np.sqrt(companies_only_new['Score'].apply(lambda x: x ** 2).sum())
        companies_only_new['Score'] = companies_only_new['Score'] / company_total_new

        # Assigning new values
        skills_only_df = skills_only_new
        companies_only_df = companies_only_new

        # Checking max iter count and relative change in total company and skill scores
        if iter_count >= max_iter:
            finished = True
        if (abs(company_total_new - company_total) + abs(skill_total_new - skill_total)) / \
            (company_total_new + skill_total_new) < tol:
            finished = True

        # Saving current skill and company total values
        skill_total = skill_total_new
        company_total = company_total_new

    return skills_only_df, companies_only_df

def remove_outliers(s):
#     return s.loc[(s<s.mean()+10*s.std()) & (s>s.mean()-10*s.std())]
    return s

def get_hybrid_score(df, cols, weights=None):
    if weights is None:
        weights = [1]*len(cols)
    return sum([weights[i]*np.log(1+(df[cols[i]] - remove_outliers(df[cols[i]]).min())/
                                  (remove_outliers(df[cols[i]]).max()-
                                   remove_outliers(df[cols[i]]).min()))
                if df[cols[i]].max() - df[cols[i]].min() > 0 else 0
                for i in range(len(cols)) ])

def get_top_n_companies_from_hits(skills_df, seed_skills, binarise=True, max_iter=20, tol=0.0001, weights=None, n=30):
    skills, companies = hits_on_companies(skills_df, seed_skills, binarise, max_iter, tol, weights)
    return companies.sort_values('Score', ascending=False).head(n).Company.values.tolist(), skills, companies

def company_emerging_skill_wrapper(skills_df, set_of_companies, start_and_end_dates):
    companies_skills = get_set_of_companies(skills_df, set_of_companies)
    companies_skill_trends = [skill_trend_features_wrapper(companies_skills,
                                                           start_and_end_dates[i][0],
                                                           start_and_end_dates[i][1],
                                                           compute_total_log_mean(
                                                                         companies_skills))
                              for i in range(len(start_and_end_dates))]
    for skill_trend_df in companies_skill_trends:
        skill_trend_df['HybridScore'] = get_hybrid_score(skill_trend_df,
                                                         ['Slope', 'Acceleration'], weights=[1, 1])
    companies_emerging_skills = [threshold_logsum_trends_simple(
        companies_skill_trends[i], col='HybridScore',
        only_positives=True,
        col_percentile_thresh=.75, pop_lower=0.001, pop_upper=0.01,
        total=get_period_of_time(compute_total_values(companies_skills),
                                 start_and_end_dates[i][0], start_and_end_dates[i][1]).sum()[0])
        for i in range(len(companies_skill_trends))]

    return companies_emerging_skills, companies_skill_trends

