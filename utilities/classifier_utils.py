from utilities.analysis_utils import *
from utilities.common_utils import *
from utilities.pandas_utils import *

def skills_to_indices(dfs_x, dfs_y, x_refcol='Skill', y_refcol='Skills'):
    """
    :param dfs_x: Dictionary mapping each time period's key to the features dataframe for the skills in that period.
    This is the output of analysis_utils.compile_all_feature_dfs
    :param dfs_y: Dictionary mapping each time period's key to the ground truth dataframe for the skills in that period.
     This is the output of reformat_y, which takes the output of
     survey_response_utils.find_majority_response_ground_truth as input.
    :param x_refcol: Reference col for x.
    :param y_refcol: Reference col for y.
    :return: Returns three dataframes and one dictionary:
    x df for skills with ground truth,
    y df for skills with ground truth,
    x df for all skills,
    dict of all indices
    All the dataframes have the common key "skillname_timeperiod" and a common index column. Bear in mind that
    there *can* be skills present in df_y_all that are absent from df_x_datapoints_with_ground_truth; these are skills
    that did not show up in any of the ads for that year, and are predicted as negative by default.

    RUN ONCE
    """
    df_x_all = pd.concat([dfs_x[k].assign(common_key=dfs_x[k][x_refcol].apply(lambda name: name + '_' + k).
                                   assign(time_period=k))
                          for k in dfs_x])
    df_y_all = pd.concat([dfs_y[k].assign(common_key=dfs_y[k][y_refcol].apply(lambda name: name + '_' + k))
                          for k in dfs_y])
    datapoints_with_ground_truth_indices = invert_dict(dict(enumerate(df_y_all.common_key.values.tolist())))
    datapoints_no_ground_truth_indices = [x for x in df_x_all.common_key.values if x not in df_y_all.common_key.values]
    datapoints_no_ground_truth_indices = {datapoints_no_ground_truth_indices[i]:
                                              len(datapoints_with_ground_truth_indices)+i
                                                    for i in range(len(datapoints_no_ground_truth_indices))}
    datapoints_all_indices = datapoints_with_ground_truth_indices.copy()
    datapoints_all_indices.update(datapoints_no_ground_truth_indices)

    df_x_datapoints_with_ground_truth = df_x_all.loc[df_x_all.common_key.apply(lambda x:
                                                         x in datapoints_with_ground_truth_indices.values())].copy()
    df_x_datapoints_with_ground_truth['common_index'] = \
                        df_x_datapoints_with_ground_truth.common_key.apply(lambda x:
                                                   datapoints_with_ground_truth_indices[x]).sort_values('common_index')
    df_y_all['common_index'] = \
                df_y_all.common_key.apply(lambda x: datapoints_with_ground_truth_indices[x]).sort_values('common_index')
    df_x_all['common_index'] = \
                df_x_all.common_key.apply(lambda x: datapoints_all_indices[x]).sort_values('common_index')

    return df_x_datapoints_with_ground_truth, df_y_all, df_x_all, datapoints_all_indices


def reformat_y(df_y_ground_truths, time_periods=TIME_PERIODS, ref_col = 'Skills'):
    """
    Prepares ground truth data for reindexing, which is itself a preprocessing step for our classification task.

    :param df_y_ground_truths: DF Output of survey_response_utils.find_majority_response_ground_truth, has
    one column per time period, where the ground truth rows have True and the rest have False.
    :param ref_col: The name of the relevant column.
    :return: A dictionary mapping each time period's key to a dataframe that has the name and whether it's ground
    truth or not (1 for positive, 0 for negative).

    RUN ONCE
    """
    result_dict = dict()
    for time_period_key in time_periods:
        new_df = df_y_ground_truths[[ref_col, time_period_key]].copy()
        new_df['row_class'] = new_df[time_period_key].apply(lambda x: 1 if x == True else 0)
        result_dict[time_period_key] = new_df[[ref_col, 'row_class']]
    return result_dict

def create_train_test_split(x_datapoints, y_datapoints, test_proportion=0.2, class_balanced=True):
    """
    Creates a training-test split which is
    :param x_datapoints: The x df
    :param y_datapoints: The y df
    :param test_proportion: Ratio of test data to all the data
    :param class_balanced: Whether to sample the test set in a balanced way from the positives and negatives (50-50) or
    to sample it fully randomly.
    :return: The training and test x dataframes, the training and test y dataframes

    RUN ONCE (all the different methods and hyperparameters should have the exact same split).
    """
    if class_balanced:
        sampled_test_y_df = y_datapoints.groupby('row_class')[['common_index', 'common_key']].\
                        apply(pd.DataFrame.sample, frac=test_proportion).reset_index().drop(columns=['level_1']).\
                        sort_values('common_index')
    else:
        sampled_test_y_df = y_datapoints.sample(frac=test_proportion).reset_index().drop(columns=['index'])
    sampled_train_y_df = y_datapoints.loc[y_datapoints.common_index.apply(lambda x:
                                         x not in sampled_test_y_df.common_index.values)].sort_values('common_index')
    sampled_test_x_df = x_datapoints.loc[x_datapoints.common_index.apply(lambda x:
                                         x in sampled_test_y_df.common_index.values)].sort_values('common_index')
    sampled_train_x_df = x_datapoints.loc[x_datapoints.common_index.apply(lambda x:
                                         x not in sampled_test_x_df.common_index.values)].sort_values('common_index')

    return sampled_train_x_df, sampled_test_x_df, sampled_train_y_df, sampled_test_y_df

def filter_out_negatives(x_datapoints, rawpop_upper_bounds, y_datapoints=None, pop_col='Job Postings Raw'):
    """
    Filters out the "base negatives" (i.e. those that don't fit in the raw popularity interval) from the x dataframe,
    then filters the y dataframe to only have the indices present in the modified x dataframe.
    :param x_datapoints: The x df
    :param rawpop_upper_bounds: The upper bound dictionary for raw popularity
    :param y_datapoints: The y df
    :param pop_col: The name of the rawpop column
    :return: Filtered x and y dataframes.

    RUN FOR EACH HYPERPARAMETER SET (I.E. UPPER BOUND)
    """
    # We first filter out the data points that have a rawpop above the threshold
    x_datapoints = x_datapoints.loc[x_datapoints.apply(lambda x: x[pop_col] < rawpop_upper_bounds[x['time_period']],
                                                                                                            axis=1)]
    # We now filter out data points that are absent from x_datapoints (either because they had a rawpop of 0
    # during the time period in question, or because their rawpop was above the quantile threshold).
    if y_datapoints is not None:
        y_datapoints = filter_out_by_common_index(x_datapoints, y_datapoints, index_col='common_index')
    # We return the x df and the y df (which could be None).
    return x_datapoints, y_datapoints

def get_indices_and_matrices(x_datapoints, y_datapoints, features_col='Features', ground_truth_col='row_class'):
    """
    Receives the aligned x and y dataframes and returns the input as a matrix, the output as a vector, and the indices
    as a list.
    :param x_datapoints: The x df
    :param y_datapoints: The y df
    :param features_col: The features column in the x df
    :param ground_truth_col: The class column in the y df
    :return: X (matrix), y (vector), indices (list)

    RUN FOR EACH RUN OF filter_out_negatives
    """
    indices = x_datapoints.common_index.values.tolist()
    X = np.vstack(x_datapoints[features_col].values)
    y = y_datapoints[ground_truth_col].values
    return X, y, indices

def fill_in_prediction_blanks(y_predicted, pred_indices, all_indices):
    """
    Fills in the blanks of the predicted y vector with negatives (0), since those are the ones removed before the
    training process.
    :param y_predicted: The predicted y vector
    :param pred_indices: The original indices of the predictions in y_predicted
    :param all_indices: All the indices
    :return: A vector that has predictions for all the indices: y_predicted for indices in pred_indices, 0 everywhere
    else.
    """
    y_result = np.zeros(len(all_indices))
    pred_current_index_pos = 0
    all_current_index_pos = 0
    while all_current_index_pos < len(all_indices):
        if pred_indices[pred_current_index_pos] == all_indices[all_current_index_pos]:
            y_result[all_current_index_pos] = y_predicted[pred_current_index_pos]
            pred_current_index_pos += 1
            all_current_index_pos += 1
        else:
            y_result[all_current_index_pos] = 0
            all_current_index_pos += 1
    return np.array(y_result)


def train_and_validate_model(data_train, data_test, rawpop_upper_bounds, validate=True):
    #
    pass