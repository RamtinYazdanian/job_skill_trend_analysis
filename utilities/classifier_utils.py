from utilities.analysis_utils import *
from utilities.common_utils import *
from utilities.pandas_utils import *
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold

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
        result_dict[time_period_key] = new_df[[ref_col, TRUTH_COL]]
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
        sampled_test_y_df = y_datapoints.groupby(TRUTH_COL)[['common_index', 'common_key']].\
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

def filter_out_negatives(x_datapoints, rawpop_upper_bounds=None, y_datapoints=None, pop_col=POP_COL,
                         return_fns=False):
    """
    Filters out the "base negatives" (i.e. those that don't exist in a time period or
    don't fit in the raw popularity interval) from the x dataframe,
    then filters the y dataframe to only have the indices present in the modified x dataframe.
    :param x_datapoints: The x df
    :param rawpop_upper_bounds: The upper bound dictionary for raw popularity
    :param y_datapoints: The y df
    :param pop_col: The name of the rawpop column
    :return: Filtered x and y dataframes.

    RUN FOR EACH HYPERPARAMETER SET (I.E. UPPER BOUND)
    """
    assert rawpop_upper_bounds is not None or y_datapoints is not None
    # We first filter out the data points that have a rawpop above the threshold
    if rawpop_upper_bounds is not None:
        x_datapoints = x_datapoints.loc[x_datapoints.apply(lambda x: x[pop_col] < rawpop_upper_bounds[x['time_period']],
                                                                                                            axis=1)]
    # We now filter out data points that are absent from x_datapoints (either because they had a rawpop of 0
    # during the time period in question, or because their rawpop was above the quantile threshold).
    fn_count = None
    if y_datapoints is not None:
        if not return_fns:
            y_datapoints = filter_out_by_common_index(x_datapoints, y_datapoints,
                                                  index_col='common_index', return_both=False)
        else:
            y_datapoints, removed_ys = filter_out_by_common_index(x_datapoints, y_datapoints,
                                                      index_col='common_index', return_both=True)
            fn_count = removed_ys[TRUTH_COL].sum()
    # We return the x df and the y df (which could be None).
    if not return_fns:
        return x_datapoints, y_datapoints
    else:
        return x_datapoints, y_datapoints, fn_count

def get_indices_and_matrices(x_datapoints, y_datapoints=None, features_col=FEATURE_COL, ground_truth_col=TRUTH_COL):
    """
    Receives the aligned x and y dataframes and returns the input as a matrix, the output as a vector, and the indices
    as a list.
    :param x_datapoints: The x df
    :param y_datapoints: The y df, can be None
    :param features_col: The features column in the x df
    :param ground_truth_col: The class column in the y df
    :return: X (matrix), y (vector), indices (list)

    RUN FOR EACH RUN OF filter_out_negatives
    """
    indices = x_datapoints.common_index.values.tolist()
    X = np.vstack(x_datapoints[features_col].values)
    if y_datapoints is not None:
        y = y_datapoints[ground_truth_col].values
    else:
        y = None
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

    RUN FOR EACH RUN OF filter_out_negatives (AFTER PREDICTION, USED TO CALCULATE FINAL F1 SCORE)
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

def predict_results_vec(clf_model, x_mat, pred_indices=None, reference_indices=None, normaliser=None):
    if normaliser is not None:
        y_pred = clf_model.predict(normaliser.transform(x_mat))
    else:
        y_pred = clf_model.predict(x_mat)
    if pred_indices is not None and reference_indices is not None:
        y_pred = fill_in_prediction_blanks(y_pred, pred_indices, reference_indices)
    return y_pred

def predict_results_df(clf_model, x_datapoints, reference_indices=None, rawpop_upper_bounds=None, normaliser=None):
    """
    Takes a model and its input, optionally filters out the default negatives, then predicts labels for the given input
    and optionally for datapoints absent from the input, and returns the result.
    :param clf_model: The classifier model
    :param x_datapoints: The dataframe containing the input data
    :param reference_indices: The indices for which a prediction is to be made. This argument exists because some data
    can be missing (because a skill never showed up) and these are datapoints that are to be predicted as negatives
    by default. Can be None.
    :param rawpop_upper_bounds: The period upper bounds for raw popularity, in order to filter out those that are
    above the threshold.
    :return: Predicted y
    """
    if rawpop_upper_bounds is not None:
        x_datapoints, discard = filter_out_negatives(x_datapoints, rawpop_upper_bounds)
    x_mat, discard, pred_indices = get_indices_and_matrices(x_datapoints, None)
    return predict_results_vec(clf_model, x_mat, pred_indices, reference_indices, normaliser)


def evaluate_results(y_pred, y_truth, type='f1'):
    if type == 'prfs':
        return precision_recall_fscore_support(y_truth, y_pred, pos_label=1, average='binary')
    elif type == 'f1':
        return f1_score(y_truth, y_pred, pos_label=1, average='binary')
    elif type == 'tpfpfn':
        conf_mat = confusion_matrix(y_truth, y_pred)
        return conf_mat[1,1], conf_mat[0,1], conf_mat[1,0]


def predict_and_evaluate_dfs(clf_model, x_datapoints, y_datapoints, rawpop_upper_bounds=None, normaliser=None):
    y_pred = predict_results_df(clf_model, x_datapoints, y_datapoints.common_index.values, rawpop_upper_bounds, normaliser)
    evaluation = evaluate_results(y_pred, y_datapoints[TRUTH_COL].values, 'f1')
    return y_pred, evaluation

def predict_and_evaluate_for_validation(clf_model, x_mat, y_vec,
                                        pred_indices=None, reference_indices=None, normaliser=None, default_fn=0):
    y_pred = predict_results_vec(clf_model, x_mat, pred_indices, reference_indices, normaliser)
    evaluation = evaluate_results(y_pred, y_vec, type='tpfpfn')
    prec = evaluation[0]/(evaluation[0]+evaluation[1])
    recall = evaluation[0]/(evaluation[0]+evaluation[2]+default_fn)
    return y_pred, 2*prec*recall/(prec+recall)

def generate_cv_folds(x_train_df, y_train_df, cv_folds=CV_FOLDS, stratified=True):
    """
    Generates cross-validation folds, consisting of a list of tuples of tuples,
    like [((trainx, trainy), (valx, valy)), ...].
    :param x_train_df: The x df
    :param y_train_df: The y df
    :param CV_FOLDS: The number of folds
    :param stratified: Whether to do it stratified or fully randomised.
    :return: [((trainx1, trainy1), (valx1, valy1)), ((trainx2, trainy2), (valx2, valy2)), ...].

    RUN ONCE (CV folds should be exactly the same for everyone)
    """
    # Only to remove 0s, which are the same no matter the hyperparameters and which we don't care about in our
    # validation F1 score.
    x_train_df, y_train_df = filter_out_negatives(x_train_df, None, y_train_df,
                                                  return_fns=False)
    if stratified:
        model = StratifiedKFold(n_splits=cv_folds, shuffle=False)
    else:
        model = KFold(n_splits=cv_folds, shuffle=False)

    resulting_indices = [index_pair for index_pair in model.split(x_train_df, y_train_df)]
    # The result. index_pair[0] has the training indices for each fold and index_pair[1] has the validation indices.
    resulting_dfs = [((x_train_df.iloc[index_pair[0]], y_train_df.iloc[index_pair[0]]),
                      (x_train_df.iloc[index_pair[1]], y_train_df.iloc[index_pair[1]]))
                      for index_pair in resulting_indices]
    return resulting_dfs


def train_logreg_model(x_df, y_df, c, rawpop_upper_bounds, normaliser=None):
    """
    Trains a logistic regression model with the given hyperparameter and rawpop upper bounds using the data in the
    two given dataframes (x and y).
    :param x_df: The x df
    :param y_df: The y df
    :param c: The regularisation hyperparameter
    :param rawpop_upper_bounds: The upper bounds dictionary
    :param normaliser: The normaliser, can be None.
    :return:
    """
    modified_x_df, modified_y_df = filter_out_negatives(x_df, rawpop_upper_bounds,
                                                        y_df,
                                                        return_fns=False)
    X_mat_train, y_vec_train, pred_indices_train = get_indices_and_matrices(modified_x_df, modified_y_df)
    if normaliser is not None:
        current_normaliser = normaliser.copy()
        X_mat_train = current_normaliser.fit_transform(X_mat_train)
    else:
        current_normaliser = None
    current_model = LogisticRegression(C=c)
    current_model.fit(X_mat_train, y_vec_train)
    return current_model, current_normaliser


def cross_validate_model(cv_data, rawpop_upper_bounds, normaliser=None):
    """

    :param cv_data: Cross-validation data, consisting of a list of tuples of tuples,
    like [((trainx, trainy), (valx, valy)), ...].
    :param rawpop_upper_bounds: Period rawpop upper bound dictionary
    :param validate: Whether cross-validation is desired. If False, this function simply works as a training wrapper.
    :return: The best trained model, its hyperparameters, and its average validation error.

    RUN FOR EACH HYPERPARAMETER SET (Logreg C and rawpop upper bound)
    """

    scores = list()
    for c in C_LIST:
        f1_score_values = list()
        for current_train, current_test in cv_data:
            current_x_df_train, current_y_df_train = current_train
            current_x_df_test, current_y_df_test = current_test
            current_model, current_normaliser = train_logreg_model(current_x_df_train, current_y_df_train, c,
                                                                   rawpop_upper_bounds, normaliser)
            current_y_pred, current_f1 = predict_and_evaluate_dfs(current_model, current_x_df_test, current_y_df_test,
                                     rawpop_upper_bounds, current_normaliser)
            f1_score_values.append(current_f1)
        scores.append((sum(f1_score_values)/CV_FOLDS))

    scores = np.array(scores)
    best_score = np.max(scores)
    best_c = C_LIST[np.argmax(scores)]
    # To get the best model trained on *all* the training data (i.e. train + validation), we concatenate the contents
    # of the first element of cv_data: x_train with x_val and y_train with y_val, and the C provided is best_c.
    best_model = train_logreg_model(pd.concat([cv_data[0][0][0], cv_data[0][1][0]], axis=0),
                                    pd.concat([cv_data[0][0][1], cv_data[0][1][1]], axis=0),
                                    best_c, rawpop_upper_bounds, normaliser)
    return best_model, best_c, best_score


def interpret_model(clf_model, feature_names):
    pass
