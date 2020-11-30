from utilities.analysis_utils import *
from utilities.common_utils import *
from utilities.pandas_utils import *
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, \
                    precision_score, accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone as sklearn_clone
from collections import Counter

from utilities.params import FEATURE_COL, TIME_PERIODS, CV_FOLDS, C_LIST, QUANTILES


def skills_to_indices(dfs_x, dfs_y, x_refcol=X_REFCOL, y_refcol=Y_REFCOL):
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

    RUN ONCE per pop method
    """
    # The sorting of both dataframes by their reference column ensures that for the different pop methods,
    # we get the exact same indices. It matters when it comes to the train/test splits and cross-validation
    # folds, since we generate the indices based on the initial order of the dataframe.
    df_x_all = pd.concat([dfs_x[k].reset_index().assign(time_period=k).
                         assign(common_key=dfs_x[k].reset_index()[x_refcol].apply(lambda name: name + '_' + k))
                          for k in dfs_x], axis=0).reset_index().drop(columns=['index']).sort_values('common_key')
    print(df_x_all.head())
    df_y_all = pd.concat([dfs_y[k].assign(common_key=dfs_y[k][y_refcol].apply(lambda name: name + '_' + k))
                          for k in dfs_y]).reset_index().drop(columns=['index']).sort_values('common_key')
    print(df_y_all.head())
    datapoints_with_ground_truth_indices = invert_dict(dict(enumerate(df_y_all.common_key.values.tolist())))
    #print(datapoints_with_ground_truth_indices)
    datapoints_no_ground_truth_indices = [x for x in df_x_all.common_key.values if x not in df_y_all.common_key.values]
    datapoints_no_ground_truth_indices = {datapoints_no_ground_truth_indices[i]:
                                              len(datapoints_with_ground_truth_indices)+i
                                                    for i in range(len(datapoints_no_ground_truth_indices))}
    datapoints_all_indices = datapoints_with_ground_truth_indices.copy()
    datapoints_all_indices.update(datapoints_no_ground_truth_indices)

    df_x_datapoints_with_ground_truth = df_x_all.loc[df_x_all.common_key.apply(lambda x:
                                                         x in datapoints_with_ground_truth_indices.keys())].copy()
    df_x_datapoints_with_ground_truth['common_index'] = \
                        df_x_datapoints_with_ground_truth.common_key.apply(lambda x:
                                                   datapoints_with_ground_truth_indices[x])
    df_y_all['common_index'] = \
                df_y_all.common_key.apply(lambda x: datapoints_with_ground_truth_indices[x])
    df_x_all['common_index'] = \
                df_x_all.common_key.apply(lambda x: datapoints_all_indices[x])

    return df_x_datapoints_with_ground_truth, df_y_all, df_x_all, datapoints_all_indices


def reformat_y(df_y_ground_truths, time_periods=TIME_PERIODS, ref_col = Y_REFCOL):
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

def create_train_test_split(x_datapoints, y_datapoints, test_proportion=0.2, class_balanced=True, random_state=1):
    """
    Creates a training-test split which is
    :param x_datapoints: The x df
    :param y_datapoints: The y df
    :param test_proportion: Ratio of test data to all the data
    :param class_balanced: Whether to sample the test set in a balanced way from the positives and negatives (50-50) or
    to sample it fully randomly.
    :return: The training and test x dataframes, the training and test y dataframes

    RUN ONCE per pop method (all the different methods and hyperparameters should have the exact same split).
    """
    if class_balanced:
        sampled_skills_y = y_datapoints[['Skills', TRUTH_COL]].groupby('Skills').apply(lambda x:
                 Counter(list(x[TRUTH_COL])).most_common()[0][0]).reset_index().rename(columns={0: TRUTH_COL})
        sampled_skills_y = sampled_skills_y.groupby(TRUTH_COL)[['Skills']].\
            apply(pd.DataFrame.sample, frac=test_proportion, random_state=random_state).reset_index()['Skills'].values
    else:
        sampled_skills_y = y_datapoints[['Skills']].drop_duplicates()\
                        .sample(frac=test_proportion)['Skills'].values

    sampled_test_y_df = y_datapoints.loc[y_datapoints['Skills'].apply(lambda x: x in sampled_skills_y)].\
                                                    reset_index().drop('index', axis=1).sort_values('common_index')
    sampled_train_y_df = y_datapoints.loc[y_datapoints.common_index.apply(lambda x:
                                         x not in sampled_test_y_df.common_index.values)].\
                                                    reset_index().drop('index', axis=1).sort_values('common_index')
    sampled_test_x_df = x_datapoints.loc[x_datapoints.common_index.apply(lambda x:
                                         x in sampled_test_y_df.common_index.values)].\
                                                    reset_index().drop('index', axis=1).sort_values('common_index')
    sampled_train_x_df = x_datapoints.loc[x_datapoints.common_index.apply(lambda x:
                                         x not in sampled_test_x_df.common_index.values)].\
                                                    reset_index().drop('index', axis=1).sort_values('common_index')

    return sampled_train_x_df, sampled_test_x_df, sampled_train_y_df, sampled_test_y_df

def pre_normalise_and_pca(sampled_train_x_df, sampled_test_x_df, sampled_train_y_df, sampled_test_y_df,
                          n_pca_features=50, n_selection_features=50,
                          pre_normaliser=None, post_normaliser=None, feature_col='tsfresh', pca_first=True):

    train_features = series_to_matrix(sampled_train_x_df[feature_col])
    test_features = series_to_matrix(sampled_test_x_df[feature_col])
    if pre_normaliser is not None:
        train_features = pre_normaliser.fit_transform(train_features)
        test_features = pre_normaliser.transform(test_features)
    if n_pca_features is not None:
        pca_model = PCA(n_components=n_pca_features)
        if n_selection_features is not None:
            feature_reduction_model = SelectKBest(score_func=mutual_info_classif, k=n_selection_features)
            if pca_first:
                train_features = pca_model.fit_transform(train_features)
                test_features = pca_model.transform(test_features)

                train_features = feature_reduction_model.fit_transform(train_features,
                                   sampled_train_y_df.loc[sampled_train_y_df.common_index.
                                    apply(lambda x: x in sampled_train_x_df.common_index.values)].row_class.values)
                test_features = feature_reduction_model.transform(test_features)
            else:
                train_features = feature_reduction_model.fit_transform(train_features,
                                       sampled_train_y_df.loc[sampled_train_y_df.common_index.
                                               apply(lambda x: x in sampled_train_x_df.common_index.values)].row_class.values)
                test_features = feature_reduction_model.transform(test_features)

                train_features = pca_model.fit_transform(train_features)
                test_features = pca_model.transform(test_features)
        else:
            feature_reduction_model = None
            train_features = pca_model.fit_transform(train_features)
            test_features = pca_model.transform(test_features)
    else:
        pca_model = None
        if n_selection_features is not None:
            feature_reduction_model = SelectKBest(score_func=mutual_info_classif, k=n_selection_features)
            train_features = feature_reduction_model.fit_transform(train_features,
                               sampled_train_y_df.loc[sampled_train_y_df.common_index.
                               apply(lambda x: x in sampled_train_x_df.common_index.values)].row_class.values)
            test_features = feature_reduction_model.transform(test_features)
        else:
            feature_reduction_model = None
    if post_normaliser is not None:
        if post_normaliser == 'unit':
            train_features = train_features / np.linalg.norm(train_features, axis=1).\
                                    reshape((train_features.shape[0], 1))
            test_features = test_features / np.linalg.norm(test_features, axis=1). \
                reshape((test_features.shape[0], 1))
            post_normaliser = None
        else:
            train_features = post_normaliser.fit_transform(train_features)
            test_features = post_normaliser.transform(test_features)
    sampled_train_x_df[feature_col+'_orig'] = sampled_train_x_df[feature_col]
    sampled_test_x_df[feature_col + '_orig'] = sampled_test_x_df[feature_col]
    sampled_train_x_df[feature_col] = train_features.tolist()
    sampled_test_x_df[feature_col] = test_features.tolist()
    return sampled_train_x_df, sampled_test_x_df, pca_model, pre_normaliser, post_normaliser, feature_reduction_model

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
    if len(pred_indices) == len(all_indices):
        return y_predicted
    # print(len(y_predicted))
    # print(len(pred_indices))
    # print(len(all_indices))
    y_result = np.zeros(len(all_indices))
    pred_current_index_pos = 0
    all_current_index_pos = 0
    while all_current_index_pos < len(all_indices):
        if pred_current_index_pos < len(pred_indices) and \
                pred_indices[pred_current_index_pos] == all_indices[all_current_index_pos]:
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
        y_pred = clf_model.predict(np.array([np.array(x) for x in x_mat]))
    # print(y_pred)
    if pred_indices is not None and reference_indices is not None:
        y_pred = fill_in_prediction_blanks(y_pred, pred_indices, reference_indices)
    return y_pred

def predict_results_df(clf_model, x_datapoints, reference_indices=None, rawpop_upper_bounds=None, normaliser=None,
                       features_col=FEATURE_COL):
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
    x_mat, discard, pred_indices = get_indices_and_matrices(x_datapoints, None, features_col=features_col)
    # print(len(pred_indices))
    return predict_results_vec(clf_model, x_mat, pred_indices, reference_indices, normaliser)


def evaluate_results(y_pred, y_truth, type='acc'):
    if type == 'prfs':
        return precision_recall_fscore_support(y_truth, y_pred, pos_label=1, average='binary')
    elif type == 'f1':
        return f1_score(y_truth, y_pred, pos_label=1, average='binary')
    elif type == 'prec':
        return precision_score(y_truth, y_pred, pos_label=1, average='binary')
    elif type == 'recall':
        return recall_score(y_truth, y_pred, pos_label=1, average='binary')
    elif type == 'acc':
        return accuracy_score(y_truth, y_pred)
    elif type == 'tpfpfn':
        conf_mat = confusion_matrix(y_truth, y_pred)
        return conf_mat[1,1], conf_mat[0,1], conf_mat[1,0]


def predict_and_evaluate_dfs(clf_model, x_datapoints, y_datapoints,
                             rawpop_upper_bounds=None, normaliser=None, eval_type='f1', features_col=FEATURE_COL,
                             return_modified_df=False, aggregate_for_skills=False):
    """
    Wrapper for predicting and evaluating the predictions.
    :param clf_model: Classifier model
    :param x_datapoints: X dataframe
    :param y_datapoints: Y dataframe
    :param rawpop_upper_bounds: Rawpop upper bound dictionary
    :param normaliser: Normaliser object
    :return: Predicted y and evaluation metric result
    """
    y_pred = predict_results_df(clf_model, x_datapoints, y_datapoints.common_index.values, rawpop_upper_bounds,
                                normaliser=normaliser, features_col=features_col)
    if not aggregate_for_skills:
        evaluation = evaluate_results(y_pred, y_datapoints[TRUTH_COL].values, eval_type)
        if not return_modified_df:
            return y_pred, evaluation
        else:
            return y_pred, evaluation, \
                        y_datapoints.assign(pred=y_pred).loc[y_pred == 1], \
                        y_datapoints.assign(pred=y_pred).loc[y_datapoints.row_class == 1],\
                        y_datapoints.assign(pred=y_pred)
    if aggregate_for_skills:
        result_df = y_datapoints.assign(pred=y_pred)
        result_df = result_df[['Skills', 'row_class', 'pred']].groupby('Skills').sum().reset_index()
        result_df['row_class'] = result_df['row_class'].apply(lambda x: 1 if x > 0 else 0)
        result_df['pred'] = result_df['pred'].apply(lambda x: 1 if x > 0 else 0)
        evaluation = evaluate_results(result_df.pred.values, result_df.row_class.values, eval_type)
        if not return_modified_df:
            return result_df.pred, evaluation
        else:
            return result_df.pred, evaluation, \
               result_df.loc[result_df.pred == 1], \
               result_df.loc[result_df.row_class == 1],\
               result_df


def generate_cv_folds(x_train_df, y_train_df, cv_folds=CV_FOLDS, stratified=True, random_state=1):
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
    x_train_df = x_train_df.sort_values('common_index').reset_index().drop(columns=['index'])
    y_train_df = y_train_df.sort_values('common_index').reset_index().drop(columns=['index'])

    if stratified:
        model = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    else:
        model = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


    resulting_indices = [index_pair for index_pair in model.split(x_train_df, None)]
    # The result. index_pair[0] has the training indices for each fold and index_pair[1] has the validation indices.
    resulting_dfs = [((x_train_df.iloc[index_pair[0]], y_train_df.iloc[index_pair[0]]),
                      (x_train_df.iloc[index_pair[1]], y_train_df.iloc[index_pair[1]]))
                      for index_pair in resulting_indices]
    return resulting_dfs


def train_logreg_model(x_df, y_df, c, rawpop_upper_bounds, normaliser=None, features_col=FEATURE_COL, penalty='l2'):
    """
    Trains a logistic regression model with the given hyperparameter and rawpop upper bounds using the data in the
    two given dataframes (x and y).
    :param x_df: The x df
    :param y_df: The y df
    :param c: The regularisation hyperparameter
    :param rawpop_upper_bounds: The upper bounds dictionary
    :param normaliser: The normaliser, can be None.
    :return: The trained model
    """
    # print()
    modified_x_df, modified_y_df = filter_out_negatives(x_df, rawpop_upper_bounds,
                                                        y_df,
                                                        return_fns=False)
    X_mat_train, y_vec_train, pred_indices_train = get_indices_and_matrices(modified_x_df, modified_y_df,
                                                                            features_col=features_col)
    if normaliser is not None:
        current_normaliser = sklearn_clone(normaliser)
        X_mat_train = current_normaliser.fit_transform(X_mat_train)
    else:
        current_normaliser = None
    current_model = LogisticRegression(C=c, max_iter=600, penalty=penalty, solver='lbfgs')
    current_model.fit(X_mat_train, y_vec_train)
    return current_model, current_normaliser

def train_svm(x_df, y_df, c, rawpop_upper_bounds, normaliser=None, features_col=FEATURE_COL, kernel='linear'):
    """
    Trains a logistic regression model with the given hyperparameter and rawpop upper bounds using the data in the
    two given dataframes (x and y).
    :param x_df: The x df
    :param y_df: The y df
    :param c: The regularisation hyperparameter
    :param rawpop_upper_bounds: The upper bounds dictionary
    :param normaliser: The normaliser, can be None.
    :return: The trained model
    """
    print()
    modified_x_df, modified_y_df = filter_out_negatives(x_df, rawpop_upper_bounds,
                                                        y_df,
                                                        return_fns=False)
    X_mat_train, y_vec_train, pred_indices_train = get_indices_and_matrices(modified_x_df, modified_y_df,
                                                                            features_col=features_col)
    if normaliser is not None:
        current_normaliser = sklearn_clone(normaliser)
        X_mat_train = current_normaliser.fit_transform(X_mat_train)
    else:
        current_normaliser = None
    # current_model = SVC(C=c, kernel=kernel)
    current_model = LinearSVC(C=c)
    current_model.fit(X_mat_train, y_vec_train)
    return current_model, current_normaliser

def train_decision_tree(x_df, y_df, c, rawpop_upper_bounds, normaliser=None, features_col=FEATURE_COL):
    """
    Trains a logistic regression model with the given hyperparameter and rawpop upper bounds using the data in the
    two given dataframes (x and y).
    :param x_df: The x df
    :param y_df: The y df
    :param c: The regularisation hyperparameter
    :param rawpop_upper_bounds: The upper bounds dictionary
    :param normaliser: The normaliser, can be None.
    :return: The trained model
    """
    print()
    modified_x_df, modified_y_df = filter_out_negatives(x_df, rawpop_upper_bounds,
                                                        y_df,
                                                        return_fns=False)
    X_mat_train, y_vec_train, pred_indices_train = get_indices_and_matrices(modified_x_df, modified_y_df,
                                                                            features_col=features_col)
    if normaliser is not None:
        current_normaliser = sklearn_clone(normaliser)
        X_mat_train = current_normaliser.fit_transform(X_mat_train)
    else:
        current_normaliser = None
    current_model = DecisionTreeClassifier(max_depth=c['max_depth'],
                                           min_samples_split=c['min_samples_split'], max_features=c['max_features'])
    current_model.fit(X_mat_train, y_vec_train)
    return current_model, current_normaliser


def cross_validate_model(cv_data, rawpop_upper_bounds, c_list,
                         normaliser=None, verbose=True, features_col=FEATURE_COL, model_to_use='logreg',
                         eval_type='f1', aggregated_skills=False):
    """

    :param cv_data: Cross-validation data, consisting of a list of tuples of tuples,
    like [((trainx, trainy), (valx, valy)), ...].
    :param rawpop_upper_bounds: Period rawpop upper bound dictionary
    :param validate: Whether cross-validation is desired. If False, this function simply works as a training wrapper.
    :return: The best trained model, its hyperparameters, and its average validation error.

    RUN FOR EACH HYPERPARAMETER SET (Logreg C and rawpop upper bound)
    """
    scores = list()
    for c in c_list:
        if verbose:
            print('C: ' + str(c) + '\n\n-----------\n\n')
        f1_score_values = list()
        for current_train, current_test in cv_data:
            current_x_df_train, current_y_df_train = current_train
            current_x_df_test, current_y_df_test = current_test
            if model_to_use == 'logreg':
                current_model, current_normaliser = train_logreg_model(current_x_df_train, current_y_df_train, c,
                                                                   rawpop_upper_bounds, normaliser=normaliser,
                                                                   features_col=features_col)
            elif model_to_use == 'dt':
                current_model, current_normaliser = train_decision_tree(current_x_df_train, current_y_df_train, c,
                                                                       rawpop_upper_bounds, normaliser=normaliser,
                                                                       features_col=features_col)
            elif model_to_use == 'svm':
                current_model, current_normaliser = train_svm(current_x_df_train, current_y_df_train, c,
                                                              rawpop_upper_bounds, normaliser=normaliser,
                                                              features_col=features_col)
            current_y_pred, current_scoring_measure = \
                                predict_and_evaluate_dfs(current_model, current_x_df_test, current_y_df_test,
                                     rawpop_upper_bounds, current_normaliser, features_col=features_col,
                                                         eval_type=eval_type, aggregate_for_skills=aggregated_skills)
            # print('Next')
            f1_score_values.append(current_scoring_measure)
        averaged_score = sum(f1_score_values)/CV_FOLDS
        if verbose:
            print('Avg score: '+str(averaged_score)+'\n\n**********\n\n')
        scores.append(averaged_score)

    scores = np.array(scores)
    best_score = np.max(scores)
    best_c = c_list[np.argmax(scores)]
    # To get the best model trained on *all* the training data (i.e. train + validation), we concatenate the contents
    # of the first element of cv_data: x_train with x_val and y_train with y_val, and the C provided is best_c.
    if model_to_use == 'logreg':
        best_model = train_logreg_model(pd.concat([cv_data[0][0][0], cv_data[0][1][0]], axis=0),
                                    pd.concat([cv_data[0][0][1], cv_data[0][1][1]], axis=0),
                                    best_c, rawpop_upper_bounds, normaliser, features_col)
    elif model_to_use == 'dt':
        best_model = train_decision_tree(pd.concat([cv_data[0][0][0], cv_data[0][1][0]], axis=0),
                                        pd.concat([cv_data[0][0][1], cv_data[0][1][1]], axis=0),
                                        best_c, rawpop_upper_bounds, normaliser, features_col)
    elif model_to_use == 'svm':
        best_model = train_svm(pd.concat([cv_data[0][0][0], cv_data[0][1][0]], axis=0),
                               pd.concat([cv_data[0][0][1], cv_data[0][1][1]], axis=0),
                               best_c, rawpop_upper_bounds, normaliser, features_col)
    return best_model, best_c, best_score

def cross_validate_with_quantile(cv_data, period_to_df, normaliser=None, verbose=True, features_col=FEATURE_COL,
                                 c_list=C_LIST, quantiles=QUANTILES, model_to_use='logreg', eval_type='f1',
                                 aggregated_skills=False):
    scores = list()
    models = list()
    cs = list()
    qs = list()
    rawpop_ubs = list()
    print(features_col)
    for q in quantiles:
        if verbose:
            print('QUANTILE: '+str(q)+'\n\n-----------\n\n')
        rawpop_upper_bound = compute_time_period_rawpop_quantile_thresholds(period_to_df, q)
        current_best_model, current_best_c, current_best_score = \
                                        cross_validate_model(cv_data, rawpop_upper_bound, normaliser=normaliser,
                                                 features_col=features_col, c_list=c_list, model_to_use=model_to_use,
                                                     eval_type=eval_type, verbose=verbose,
                                                             aggregated_skills=aggregated_skills)
        rawpop_ubs.append(rawpop_upper_bound)
        scores.append(current_best_score)
        models.append(current_best_model)
        cs.append(current_best_c)
        qs.append(q)
    best_score = np.max(scores)
    best_model = models[np.argmax(scores)]
    best_c = cs[np.argmax(scores)]
    best_q = qs[np.argmax(scores)]
    best_rawpop_upper_bound = rawpop_ubs[np.argmax(scores)]

    return best_model, best_c, best_q, best_score, best_rawpop_upper_bound

def normalise_pca_and_cross_validate_wrap(df_x_datapoints_with_ground_truth, df_y_all,
                          period_to_df, pca_feature_counts_list, n_selection_features=None, test_proportion=0.2,
                          pre_pca_norms = None, pre_clf_norms = None,
                           verbose=True, features_col='tsfresh',
                             c_list=C_LIST, quantiles=QUANTILES, model_to_use='logreg', eval_type='f1',
                                          random_state=1):

    print('Random state: ' + str(random_state))

    pre_norm_list = list()
    pcas_list = list()
    scores = list()
    models = list()
    cs = list()
    qs = list()
    rawpop_ubs = list()
    train_list = list()
    test_list = list()

    for pre_norm in pre_pca_norms:
        for post_norm in pre_clf_norms:
            for n_pca_features in pca_feature_counts_list:
                sampled_train_x_df, sampled_test_x_df, sampled_train_y_df, sampled_test_y_df = \
                    create_train_test_split(df_x_datapoints_with_ground_truth, df_y_all,
                                            test_proportion=test_proportion, class_balanced=True,
                                                    random_state=random_state)

                sampled_train_x_df, sampled_test_x_df, pca_model, pre_normaliser, throwaway, feature_selection_model = \
                    pre_normalise_and_pca(sampled_train_x_df, sampled_test_x_df, sampled_train_y_df, sampled_test_y_df,
                                          n_pca_features=n_pca_features, n_selection_features=n_selection_features,
                                          pre_normaliser=sklearn_clone(pre_norm),
                                          post_normaliser=None, pca_first=True)

                cv_data = generate_cv_folds(sampled_train_x_df, sampled_train_y_df, stratified=False)

                normaliser = sklearn_clone(post_norm)

                current_model, current_c, current_q, current_score, current_rawpop_ub = \
                    cross_validate_with_quantile(cv_data, period_to_df, normaliser=normaliser,
                                     verbose=verbose, features_col=features_col, c_list=c_list, quantiles=quantiles,
                                                 model_to_use=model_to_use, eval_type=eval_type)

                print('Model with: ')
                print('Pre-norm:')
                print(pre_norm)
                print('Post-norm:')
                print(post_norm)
                print('Results: ' + eval_type + ' = ' + str(current_score))

                pre_norm_list.append(pre_normaliser)
                pcas_list.append([pca_model, feature_selection_model])
                rawpop_ubs.append(current_rawpop_ub)
                scores.append(current_score)
                models.append(current_model)
                cs.append(current_c)
                qs.append(current_q)
                train_list.append((sampled_train_x_df, sampled_train_y_df))
                test_list.append((sampled_test_x_df, sampled_test_y_df))

    best_index = np.argmax(scores)
    print('**************** Best: ')
    print('Model with: ')
    print('Pre-norm:')
    print(pre_norm_list[best_index])
    print('Post-norm:')
    print(models[best_index][1])
    print('PCA model:')
    print(pcas_list[best_index][0])
    print('Results: ' + eval_type + ' = ' + str(scores[best_index]))
    print('Evaluation on train/test: ')
    print('Test: ')
    print(predict_and_evaluate_dfs(models[best_index][0], test_list[best_index][0], test_list[best_index][1],
                                   rawpop_ubs[best_index], models[best_index][1], eval_type='prfs',
                                   features_col=features_col,
                                   return_modified_df=True, aggregate_for_skills=True))
    print('Train: ')
    print(predict_and_evaluate_dfs(models[best_index][0], train_list[best_index][0], train_list[best_index][1],
                                    rawpop_ubs[best_index], models[best_index][1], eval_type='prfs',
                                            features_col=features_col,
                                                    return_modified_df=True, aggregate_for_skills=True))
    print('------------------------------------')
    return {i: (models[i], cs[i], qs[i], scores[i], rawpop_ubs[i], pre_norm_list[i], pcas_list[i])
            for i in range(len(models))}, best_index


def test_eval_for_wrapped_results(df_x_datapoints_with_ground_truth, df_y_all,
        results_dict, best_index, random_state, already_has_feature_selection=False,
                                  n_selection_features=90, test_proportion=0.2):
    wrapped_results = results_dict[best_index]
    best_model = wrapped_results[0]
    pca_model = wrapped_results[6]
    rawpop_upper_bound_dict = wrapped_results[4]
    prenormaliser = wrapped_results[5]

    sampled_train_x_df, sampled_test_x_df, sampled_train_y_df, sampled_test_y_df = \
        create_train_test_split(df_x_datapoints_with_ground_truth, df_y_all,
                                test_proportion=test_proportion, class_balanced=True,
                                random_state=random_state)
    train_features = prenormaliser.transform(series_to_matrix(sampled_train_x_df['tsfresh']))
    test_features = prenormaliser.transform(series_to_matrix(sampled_test_x_df['tsfresh']))
    if not already_has_feature_selection:
        feature_reduction_model = SelectKBest(score_func=mutual_info_classif, k=n_selection_features)
        train_features = pca_model.fit_transform(train_features)
        test_features = pca_model.transform(test_features)

        train_features = feature_reduction_model.fit_transform(train_features,
                   sampled_train_y_df.loc[sampled_train_y_df.common_index.
                   apply(lambda x: x in sampled_train_x_df.common_index.values)].row_class.values)
        test_features = feature_reduction_model.transform(test_features)
    else:
        feature_reduction_model = pca_model[1]
        pca_model = pca_model[0]
        train_features = pca_model.fit_transform(train_features)
        test_features = pca_model.transform(test_features)

        train_features = feature_reduction_model.fit_transform(train_features,
                   sampled_train_y_df.loc[sampled_train_y_df.common_index.
                   apply(lambda x: x in sampled_train_x_df.common_index.values)].row_class.values)
        test_features = feature_reduction_model.transform(test_features)

    sampled_train_x_df['tsfresh'] = train_features.tolist()
    sampled_test_x_df['tsfresh'] = test_features.tolist()

    print('**********Test set, not aggregated')

    test_y_pred, test_y_eval, test_pred_positive_df, test_actual_positive_df, test_all_pred = \
        predict_and_evaluate_dfs(best_model[0], sampled_test_x_df, sampled_test_y_df,
                                 rawpop_upper_bounds=rawpop_upper_bound_dict,
                                 features_col='tsfresh',
                                 normaliser=best_model[1], eval_type='prfs', return_modified_df=True)

    print(test_y_pred, test_y_eval, test_pred_positive_df, test_actual_positive_df)

    print('-----------Baseline-----------')
    print(
        evaluate_results(np.array([1] * sampled_test_y_df.shape[0]), sampled_test_y_df[TRUTH_COL].values, type='prfs'))

    print('**********Test set, aggregated')

    test_y_pred, test_y_eval, test_pred_positive_df, test_actual_positive_df, test_all_pred = \
        predict_and_evaluate_dfs(best_model[0], sampled_test_x_df, sampled_test_y_df,
                                 rawpop_upper_bounds=rawpop_upper_bound_dict,
                                 features_col='tsfresh',
                                 normaliser=best_model[1], eval_type='prfs', return_modified_df=True,
                                 aggregate_for_skills=True)

    print(test_y_pred, test_y_eval, test_pred_positive_df, test_actual_positive_df)

    print('-----------Baseline-----------')
    agg_test_gt = sampled_test_y_df[['Skills', 'row_class']].groupby('Skills'). \
        apply(lambda x: 1 if x['row_class'].sum() > 0 else 0) \
        .values
    print(evaluate_results(np.array([1] * len(sampled_test_y_df.Skills.unique())), agg_test_gt
                           , type='prfs'))

    print('**********Training set, not aggregated')

    train_y_pred, train_y_eval, train_pred_positive_df, train_actual_positive_df, train_all_pred = \
        predict_and_evaluate_dfs(best_model[0], sampled_train_x_df, sampled_train_y_df,
                                 rawpop_upper_bounds=rawpop_upper_bound_dict,
                                 normaliser=best_model[1], eval_type='prfs', features_col='tsfresh',
                                 return_modified_df=True)

    print(train_y_pred, train_y_eval, train_pred_positive_df, train_actual_positive_df)

    print('-----------Baseline-----------')
    print(
        evaluate_results(np.array([1] * sampled_train_y_df.shape[0]), sampled_train_y_df[TRUTH_COL].values, type='prfs'))

    print('**********Training set, aggregated')

    train_y_pred, train_y_eval, train_pred_positive_df, train_actual_positive_df, train_all_pred = \
        predict_and_evaluate_dfs(best_model[0], sampled_train_x_df, sampled_train_y_df,
                                 rawpop_upper_bounds=rawpop_upper_bound_dict,
                                 normaliser=best_model[1], eval_type='prfs', features_col='tsfresh',
                                 return_modified_df=True, aggregate_for_skills=True)

    print(train_y_pred, train_y_eval, train_pred_positive_df, train_actual_positive_df)

    print('-----------Baseline-----------')
    agg_test_gt = sampled_train_y_df[['Skills', 'row_class']].groupby('Skills'). \
        apply(lambda x: 1 if x['row_class'].sum() > 0 else 0) \
        .values
    print(evaluate_results(np.array([1] * len(sampled_train_y_df.Skills.unique())), agg_test_gt
                           , type='prfs'))


def interpret_model(clf_model, feature_names, feature_scores=None,
                    pca_model=None, n_features=None, dataframes=None, normaliser=None):
    if feature_scores is None:
        scores = [-clf_model.intercept_[0]] + clf_model.coef_.flatten().tolist()
    else:
        scores = [-clf_model.intercept_[0]] + feature_scores.flatten().tolist()
    if pca_model is None:
        names = ['s_min']+feature_names
        interpretation_df = pd.DataFrame({'Name': names,
                                          'Score': scores})
    else:
        names = ['s_min']+list(range(len(clf_model.coef_.flatten())))
        og_names = feature_names
        features_list = ['None'] + [[(og_names[i], pca_model.components_[j,i])
                                        for i in (np.argsort(np.abs(pca_model.components_[j,:]).flatten()))[-20:]][::-1]
                                           for j in range(pca_model.components_.shape[0])]
        interpretation_df = pd.DataFrame({'Name': names,
                          'Score': scores,
                          'Features': features_list})

    if dataframes is not None:
        if normaliser is None:
            normaliser = StandardScaler(with_mean=False, with_std=False)
            normaliser.fit(series_to_matrix(dataframes[0]))
        interpretation_df['train_median'] = \
            [0] + np.median(normaliser.transform(series_to_matrix(dataframes[0])), axis=0).flatten().tolist()
        interpretation_df['test_median'] = \
            [0] + np.median(normaliser.transform(series_to_matrix(dataframes[1])), axis=0).flatten().tolist()
        interpretation_df['train_std'] = \
            [0] + (np.quantile(normaliser.transform(series_to_matrix(dataframes[0])), q=0.75, axis=0) -
                   np.quantile(normaliser.transform(series_to_matrix(dataframes[0])), q=0.25, axis=0)).flatten().tolist()
        interpretation_df['test_std'] = \
            [0] + (np.quantile(normaliser.transform(series_to_matrix(dataframes[1])), q=0.75, axis=0) -
                   np.quantile(normaliser.transform(series_to_matrix(dataframes[1])), q=0.25, axis=0)).flatten().tolist()

    interpretation_df = interpretation_df.sort_values('Score', ascending=False)
    if n_features is None:
        return interpretation_df
    else:
        return pd.concat([interpretation_df.loc[interpretation_df.Name == 's_min'],
                          interpretation_df.head(n_features),
                          interpretation_df.tail(n_features)], axis=0). \
            drop_duplicates().sort_values('Score', ascending=False)


def get_error_rate_for_each_period(clf_model, x_datapoints, y_datapoints, time_periods,
                                   rawpop_upper_bounds=None, normaliser=None, eval_type='f1'):
    results = dict()
    for period in time_periods:
        results[period] = predict_and_evaluate_dfs(clf_model,
                                   x_datapoints.loc[x_datapoints[X_REFCOL].apply(lambda sp: period in sp)],
                                   y_datapoints.loc[y_datapoints[Y_REFCOL].apply(lambda sp: period in sp)],
                                   rawpop_upper_bounds, normaliser, eval_type)[1]
    return results