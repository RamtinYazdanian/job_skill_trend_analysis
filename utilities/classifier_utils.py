from utilities.analysis_utils import *
from utilities.common_utils import *
from utilities.pandas_utils import *
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, \
                    precision_score, accuracy_score, recall_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, \
    mutual_info_classif, mutual_info_regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone as sklearn_clone
from collections import Counter
import time

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
    dfs_x = {k: dfs_x[k].reset_index().sort_values(x_refcol) for k in dfs_x}
    dfs_y = {k: dfs_y[k].reset_index().sort_values(y_refcol) for k in dfs_y}
    df_x_all = pd.concat([dfs_x[k].assign(time_period=k).
                         assign(common_key=dfs_x[k][x_refcol].apply(lambda name: name + '_' + k))
                          for k in dfs_x], axis=0).reset_index().drop(columns=['index']).sort_values('common_key')
    print(df_x_all.head())
    df_y_all = pd.concat([dfs_y[k].assign(common_key=dfs_y[k][y_refcol].apply(lambda name: name + '_' + k))
                          for k in dfs_y]).reset_index().drop(columns=['index']).sort_values('common_key')
    print(df_y_all.head())
    datapoints_with_ground_truth_indices = invert_dict(dict(enumerate(df_y_all.common_key.values.tolist())))
    #print(datapoints_with_ground_truth_indices)
    datapoints_no_ground_truth_indices = [x for x in df_x_all.common_key.values if x not in df_y_all.common_key.values]
    print(datapoints_no_ground_truth_indices)
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


def continuous_reformat_y(aficf_scores, time_periods=TIME_PERIODS, skill_col='TagName', ref_col='score_diff'):
    result_dict = dict()
    for time_period_key in time_periods:
        new_df = aficf_scores.loc[aficf_scores.year == time_period_key.split('-')[1],
                                  [ref_col, skill_col]].copy()
        new_df = new_df.rename(columns={skill_col: Y_REFCOL, ref_col: TRUTH_COL})
        result_dict[time_period_key] = new_df
    return result_dict

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

def create_train_test_split(x_datapoints, y_datapoints, test_proportion=0.2, class_balanced=True,
                            based_on_year=False, also_based_on_skill=False, random_state=1, truth_col=TRUTH_COL):
    """
    Creates a training-test split which is
    :param x_datapoints: The x df
    :param y_datapoints: The y df
    :param test_proportion: Ratio of test data to all the data
    :param class_balanced: Whether to sample the test set in a balanced way from the positives and negatives
    (such that the pos/neg ratio is the same in the training and test sets) or to sample it fully randomly.
    :return: The training and test x dataframes, the training and test y dataframes

    RUN ONCE per pop method (all the different methods and hyperparameters should have the exact same split).
    """

    print('BEFORE THE SPLIT:')
    print(y_datapoints.common_index.sum())
    print(y_datapoints.shape)

    print('Creating train/test split')
    if not based_on_year:
        if class_balanced:
            sampled_test_skills = y_datapoints[['Skills', truth_col]].groupby('Skills').apply(lambda x:
                     Counter(list(x[truth_col])).most_common()[0][0]).reset_index().rename(columns={0: truth_col})
            sampled_test_skills = sampled_test_skills.groupby(truth_col)[['Skills']].\
                apply(pd.DataFrame.sample, frac=test_proportion, random_state=random_state).\
                                    reset_index()['Skills'].values
        else:
            sampled_test_skills = y_datapoints[['Skills']].drop_duplicates()\
                            .sample(frac=test_proportion, random_state=random_state)['Skills'].values

        sampled_test_y_df = y_datapoints.loc[y_datapoints['Skills'].apply(lambda x: x in sampled_test_skills)].\
                                                    reset_index().drop('index', axis=1).sort_values('common_index')
        print(sampled_test_y_df.common_index.sum())
    else:
        unique_periods = y_datapoints.common_key.apply(lambda x: x.split('_')[1]).sort_values().unique()
        last_period = unique_periods[-1]
        print(last_period)

        if also_based_on_skill:
            sampled_test_skills = y_datapoints.loc[y_datapoints.common_key.apply(
                    lambda x: x.split('_')[1] == last_period)][['common_key', truth_col]]

            sampled_test_skills = sampled_test_skills.groupby(truth_col)[['common_key']]. \
                    apply(pd.DataFrame.sample, frac=test_proportion, random_state=random_state). \
                        reset_index()['common_key'].values.tolist()

            all_the_other_skills = y_datapoints.loc[y_datapoints.common_key.apply(
                lambda x: x.split('_')[1] != last_period)]
            all_the_other_skills = all_the_other_skills.loc[all_the_other_skills.common_key.apply(lambda x:
                                  x.split('_')[0]+last_period not in sampled_test_skills)].common_key.values.tolist()

            all_common_keys_to_keep = set(sampled_test_skills).union(set(all_the_other_skills))
            y_datapoints = y_datapoints.copy().\
                            loc[y_datapoints.common_key.apply(lambda x: x in all_common_keys_to_keep)]
            x_datapoints = x_datapoints.copy().\
                            loc[x_datapoints.common_key.apply(lambda x: x in all_common_keys_to_keep)]

        sampled_test_y_df = y_datapoints.loc[y_datapoints.common_key.apply(
                                    lambda x: x.split('_')[1] == last_period)].reset_index().\
                                        drop('index', axis=1).sort_values('common_index')
        print('Data on sampled test Y')
        print(sampled_test_y_df.common_index.sum())
        print(sampled_test_y_df.shape)
        print('Data on the whole thing')
        print(y_datapoints.common_index.sum())
        print(y_datapoints.shape)


    sampled_train_y_df = y_datapoints.loc[y_datapoints.common_index.apply(lambda x:
                                         x not in sampled_test_y_df.common_index.values)].\
                                                    reset_index().drop('index', axis=1).sort_values('common_index')
    sampled_test_x_df = x_datapoints.loc[x_datapoints.common_index.apply(lambda x:
                                         x in sampled_test_y_df.common_index.values)].\
                                                    reset_index().drop('index', axis=1).sort_values('common_index')
    print(sampled_test_x_df.common_index.sum())
    sampled_train_x_df = x_datapoints.loc[x_datapoints.common_index.apply(lambda x:
                                         x not in sampled_test_x_df.common_index.values)].\
                                                    reset_index().drop('index', axis=1).sort_values('common_index')

    return sampled_train_x_df, sampled_test_x_df, sampled_train_y_df, sampled_test_y_df

def pre_normalise_and_pca(sampled_train_x_df, sampled_test_x_df, sampled_train_y_df, sampled_test_y_df,
                          n_pca_features=50, n_selection_features=None, is_clf=True,
                          pre_normaliser=None, post_normaliser=None, feature_col='tsfresh', pca_first=True):

    train_features = series_to_matrix(sampled_train_x_df[feature_col])
    test_features = series_to_matrix(sampled_test_x_df[feature_col])
    if pre_normaliser is not None:
        train_features = pre_normaliser.fit_transform(train_features)
        test_features = pre_normaliser.transform(test_features)
    if n_pca_features is not None:
        pca_model = PCA(n_components=n_pca_features)
        if n_selection_features is not None:
            if is_clf:
                feature_reduction_model = SelectKBest(score_func=f_classif, k=n_selection_features)
            else:
                feature_reduction_model = SelectKBest(score_func=f_regression, k=n_selection_features)
            if pca_first:
                train_features = pca_model.fit_transform(train_features)
                test_features = pca_model.transform(test_features)

                train_features = feature_reduction_model.fit_transform(train_features,
                                   sampled_train_y_df.loc[sampled_train_y_df.common_index.
                                     apply(lambda x: x in sampled_train_x_df.common_index.values)].
                                                                       row_class.values.tolist())
                test_features = feature_reduction_model.transform(test_features)
            else:
                train_features = feature_reduction_model.fit_transform(train_features,
                               np.array(sampled_train_y_df.loc[sampled_train_y_df.common_index.
                                 apply(lambda x: x in sampled_train_x_df.common_index.values)].
                                        row_class.values.tolist()))
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
            if is_clf:
                feature_reduction_model = SelectKBest(score_func=mutual_info_classif, k=n_selection_features)
            else:
                feature_reduction_model = SelectKBest(score_func=mutual_info_regression, k=n_selection_features)
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

def predict_results_vec(clf_model, x_mat, pred_indices=None, reference_indices=None, normaliser=None,
                        return_proba=False):
    if not return_proba:
        if normaliser is not None:
            y_pred = clf_model.predict(normaliser.transform(x_mat))
        else:
            y_pred = clf_model.predict(np.array([np.array(x) for x in x_mat]))
    else:
        if normaliser is not None:
            y_pred = clf_model.predict_proba(normaliser.transform(x_mat))
        else:
            y_pred = clf_model.predict_proba(np.array([np.array(x) for x in x_mat]))
    if pred_indices is not None and reference_indices is not None:
        y_pred = fill_in_prediction_blanks(y_pred, pred_indices, reference_indices)

    return y_pred

def predict_results_df(clf_or_reg_model, x_datapoints, y_datapoints, rawpop_upper_bounds=None, normaliser=None,
                       features_col=FEATURE_COL, return_proba=False):
    """
    Takes a model and its input, optionally filters out the default negatives, then predicts labels for the given input
    and optionally for datapoints absent from the input, and returns the result.
    :param clf_or_reg_model: The classifier model
    :param x_datapoints: The dataframe containing the input data
    :param reference_indices: The indices for which a prediction is to be made. This argument exists because some data
    can be missing (because a skill never showed up) and these are datapoints that are to be predicted as negatives
    by default. Can be None.
    :param rawpop_upper_bounds: The period upper bounds for raw popularity, in order to filter out those that are
    above the threshold.
    :return: Predicted y
    """

    if rawpop_upper_bounds is not None:
        x_datapoints, useless = filter_out_negatives(x_datapoints, rawpop_upper_bounds)

    x_mat, discard, pred_indices = get_indices_and_matrices(x_datapoints, None, features_col=features_col)

    # print(len(pred_indices))
    return predict_results_vec(clf_or_reg_model, x_mat, pred_indices, y_datapoints.common_index.values, normaliser,
                               return_proba=return_proba)


def evaluate_results(y_pred, y_truth, eval_type='acc'):
    if eval_type == 'prfs':
        return precision_recall_fscore_support(y_truth, y_pred, pos_label=1, average='binary')
    elif eval_type == 'f1':
        return f1_score(y_truth, y_pred, pos_label=1, average='binary')
    elif eval_type == 'prec':
        return precision_score(y_truth, y_pred, pos_label=1, average='binary')
    elif eval_type == 'recall':
        return recall_score(y_truth, y_pred, pos_label=1, average='binary')
    elif eval_type == 'acc':
        return accuracy_score(y_truth, y_pred)
    elif eval_type == 'tpfpfn':
        conf_mat = confusion_matrix(y_truth, y_pred)
        return conf_mat[1,1], conf_mat[0,1], conf_mat[1,0]
    elif eval_type == 'rmse':
        return mean_squared_error(y_truth, y_pred, squared=False)


def predict_and_binarise_to_evaluate(clf_or_reg_model, x_datapoints, y_datapoints,
                                         eligible_skills, rank_q,
                                            normaliser=None, eval_type='prfs', skills_to_keep=None,
                                                features_col=FEATURE_COL,
                                                    return_modified_df=False):
    x_datapoints = x_datapoints.sort_values('common_index')
    y_datapoints = y_datapoints.sort_values('common_index')
    y_pred = predict_results_df(clf_or_reg_model, x_datapoints, y_datapoints, None,
                                normaliser=normaliser, features_col=features_col, return_proba=False)
    y_datapoints = y_datapoints.assign(cont_pred=y_pred)
    skills_predicted_positive = y_datapoints.loc[(y_datapoints.Skills.apply(lambda x: x in eligible_skills)) &
                                     (y_datapoints.cont_pred >= y_datapoints.cont_pred.quantile(rank_q))].\
                                                        Skills.values
    y_datapoints['pred'] = y_datapoints.Skills.apply(lambda x: 1 if x in skills_predicted_positive else 0)
    if skills_to_keep is not None:
        y_datapoints = y_datapoints.loc[y_datapoints.common_key.apply(lambda x: x in skills_to_keep)]
    y_pred = y_datapoints.pred.values
    y_truth = y_datapoints.row_class.values
    evaluation = evaluate_results(y_pred, y_truth, eval_type)
    if not return_modified_df:
        return y_pred, evaluation
    else:
        return y_pred, evaluation, \
               y_datapoints.loc[y_datapoints.pred == 1], \
               y_datapoints.loc[y_datapoints.row_class == 1], \
               y_datapoints

def predict_and_evaluate_dfs(clf_or_reg_model, x_datapoints, y_datapoints,
                             rawpop_upper_bounds=None, normaliser=None, eval_type='f1', skills_to_keep = None,
                             features_col=FEATURE_COL,
                             return_modified_df=False, aggregate_for_skills=False, is_bin=True):
    """
    Wrapper for predicting and evaluating the predictions.
    :param clf_or_reg_model: Classifier model
    :param x_datapoints: X dataframe
    :param y_datapoints: Y dataframe
    :param rawpop_upper_bounds: Rawpop upper bound dictionary
    :param normaliser: Normaliser object
    :return: Predicted y and evaluation metric result
    """

    x_datapoints = x_datapoints.sort_values('common_index')
    y_datapoints = y_datapoints.sort_values('common_index')

    if is_bin:
        y_pred = predict_results_df(clf_or_reg_model, x_datapoints, y_datapoints, rawpop_upper_bounds,
                                    normaliser=normaliser, features_col=features_col, return_proba=False)
        if skills_to_keep is not None:
            y_df_pred = y_datapoints.assign(pred=y_pred)
            y_df_pred = y_df_pred.loc[y_df_pred.common_key.apply(lambda x: x in skills_to_keep)]
        else:
            y_df_pred = y_datapoints.assign(pred=y_pred)
        y_pred = y_df_pred.pred.values
        y_truth = y_df_pred.row_class.values
        if not aggregate_for_skills:
            evaluation = evaluate_results(y_pred, y_truth, eval_type)
            if not return_modified_df:
                return y_pred, evaluation
            else:
                return y_pred, evaluation, \
                            y_df_pred.loc[y_pred == 1], \
                            y_df_pred.loc[y_truth == 1],\
                            y_df_pred
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
    else:
        y_pred = predict_results_df(clf_or_reg_model, x_datapoints, y_datapoints, None,
                                    normaliser=normaliser, features_col=features_col, return_proba=False)
        evaluation = evaluate_results(y_pred, y_datapoints[TRUTH_COL].values, eval_type)
        if not return_modified_df:
            return y_pred, evaluation
        else:
            return y_pred, evaluation, y_datapoints.assign(pred=y_pred)


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
    current_model = LogisticRegression(C=c, max_iter=5000, penalty=penalty, solver='lbfgs')
    current_model.fit(X_mat_train, y_vec_train)
    return current_model, current_normaliser

def train_linreg(x_df, y_df, c, normaliser=None, features_col=FEATURE_COL):
    """
    Trains a linear regression model (for continuous ground truth)
    with the given hyperparameter and rawpop upper bounds using the data in the
    two given dataframes (x and y).
    :param x_df: The x df
    :param y_df: The y df
    :param c: The regularisation hyperparameter
    :param normaliser: The normaliser, can be None.
    :return: The trained model
    """
    # print()

    X_mat_train, y_vec_train, pred_indices_train = get_indices_and_matrices(x_df, y_df,
                                                                            features_col=features_col)
    if normaliser is not None:
        current_normaliser = sklearn_clone(normaliser)
        X_mat_train = current_normaliser.fit_transform(X_mat_train)
    else:
        current_normaliser = None
    current_model = Ridge(alpha=c)
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
    current_model = LinearSVC(C=c, max_iter=3000)
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
                         eval_type='f1', aggregated_skills=False, is_bin=True):
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
            elif model_to_use == 'linreg':
                current_model, current_normaliser = train_linreg(current_x_df_train, current_y_df_train, c,
                                                              normaliser=normaliser,
                                                              features_col=features_col)
            current_y_pred, current_scoring_measure = \
                                predict_and_evaluate_dfs(current_model, current_x_df_test, current_y_df_test,
                                     rawpop_upper_bounds, current_normaliser, features_col=features_col,
                                             eval_type=eval_type, aggregate_for_skills=aggregated_skills,
                                                 is_bin=is_bin)

            # TODO write train_linreg, add rmse evaluation to evaluate_results
            # print('Next')
            f1_score_values.append(current_scoring_measure)
        averaged_score = sum(f1_score_values)/CV_FOLDS
        if verbose:
            print('Avg score: '+str(averaged_score)+'\n\n**********\n\n')
        scores.append(averaged_score)

    scores = np.array(scores)
    if eval_type == 'rmse':
        scores = -scores
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
    elif model_to_use == 'linreg':
        best_model = train_linreg(pd.concat([cv_data[0][0][0], cv_data[0][1][0]], axis=0),
                               pd.concat([cv_data[0][0][1], cv_data[0][1][1]], axis=0),
                               best_c, normaliser, features_col)
    return best_model, best_c, best_score

def cross_validate_with_quantile(cv_data, period_to_df, normaliser=None, verbose=True, features_col=FEATURE_COL,
                                 c_list=C_LIST, quantiles=QUANTILES, model_to_use='logreg', eval_type='f1',
                                 aggregated_skills=False, is_bin=True):
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
                                                             aggregated_skills=aggregated_skills,
                                                                is_bin=is_bin)
        rawpop_ubs.append(rawpop_upper_bound)
        scores.append(current_best_score)
        models.append(current_best_model)
        cs.append(current_best_c)
        qs.append(q)
    scores = np.array(scores)
    if eval_type == 'rmse':
        scores = -scores
    best_score = np.max(scores)
    best_model = models[np.argmax(scores)]
    best_c = cs[np.argmax(scores)]
    best_q = qs[np.argmax(scores)]
    best_rawpop_upper_bound = rawpop_ubs[np.argmax(scores)]

    return best_model, best_c, best_q, best_score, best_rawpop_upper_bound

def normalise_pca_and_cross_validate_wrap(df_x_datapoints_with_ground_truth, df_y_all,
                          period_to_df, pca_feature_counts_list, n_selection_features=None, test_proportion=0.2,
                          based_on_year=False, also_based_on_skill=False,
                          pre_pca_norms = None, pre_clf_norms = None,
                          verbose=True, features_col='tsfresh',
                          c_list=C_LIST, quantiles=QUANTILES, model_to_use='logreg', eval_type='f1',
                          random_state=1, common_keys_to_keep=None, is_bin=True):

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

    if not is_bin:
        eval_type = 'rmse'
        output_eval_type = 'rmse'
    else:
        output_eval_type = 'prfs'

    if common_keys_to_keep is not None:
        df_x_datapoints_with_ground_truth = df_x_datapoints_with_ground_truth.loc[
            df_x_datapoints_with_ground_truth.common_key.apply(lambda x: x in common_keys_to_keep)
        ]
        df_y_all = df_y_all.loc[
            df_y_all.common_key.apply(lambda x: x in common_keys_to_keep)
        ]

    n_orig_features = len(df_x_datapoints_with_ground_truth[features_col].values[0])
    if pca_feature_counts_list is None or pca_feature_counts_list == []:
        pca_feature_counts_list = [n_orig_features]
    if n_selection_features is None or n_selection_features == []:
        n_selection_features = [n_orig_features]
    pca_feature_counts_list = [x if x <= n_orig_features else n_orig_features for x in pca_feature_counts_list]
    pca_feature_counts_list = sorted(list(set(pca_feature_counts_list)))

    for pre_norm in pre_pca_norms:
        for post_norm in pre_clf_norms:
            for n_features_chosen in n_selection_features:
                for n_pca_features in pca_feature_counts_list:
                    class_balanced = (not based_on_year) and is_bin
                    truth_col = TRUTH_COL if is_bin else AUX_TRUTH_COL
                    sampled_train_x_df, sampled_test_x_df, sampled_train_y_df, sampled_test_y_df = \
                        create_train_test_split(df_x_datapoints_with_ground_truth, df_y_all,
                                                test_proportion=test_proportion,
                                                based_on_year=based_on_year, also_based_on_skill=also_based_on_skill,
                                                class_balanced=class_balanced,
                                                random_state=random_state, truth_col=truth_col)
                    current_prenorm = None
                    if pre_norm is not None:
                        current_prenorm = sklearn_clone(pre_norm)
                    sampled_train_x_df, sampled_test_x_df, pca_model, pre_normaliser, throwaway, feature_selection_model = \
                        pre_normalise_and_pca(sampled_train_x_df, sampled_test_x_df, sampled_train_y_df, sampled_test_y_df,
                                              n_pca_features=n_pca_features, n_selection_features=n_features_chosen,
                                              pre_normaliser=current_prenorm, is_clf=is_bin,
                                              post_normaliser=None, pca_first=False, feature_col=features_col)

                    cv_data = generate_cv_folds(sampled_train_x_df, sampled_train_y_df, stratified=False)

                    normaliser = None
                    if post_norm is not None:
                        normaliser = sklearn_clone(post_norm)

                    current_model, current_c, current_q, current_score, current_rawpop_ub = \
                        cross_validate_with_quantile(cv_data, period_to_df, normaliser=normaliser,
                                     verbose=verbose, features_col=features_col, c_list=c_list, quantiles=quantiles,
                                             model_to_use=model_to_use, eval_type=eval_type, is_bin=is_bin)

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

    scores = np.array(scores)
    if eval_type == 'rmse':
        scores = -scores
    best_index = np.argmax(scores)
    print('**************** Best: ')
    print('Model with: ')
    print('Pre-norm:')
    print(pre_norm_list[best_index])
    print('Post-norm:')
    print(models[best_index][1])
    print('PCA model:')
    print(pcas_list[best_index])
    print('Results: ' + eval_type + ' = ' + str(scores[best_index]))
    print('Evaluation on train/test: ')
    print('Test: ')
    print(predict_and_evaluate_dfs(models[best_index][0], test_list[best_index][0], test_list[best_index][1],
                                   rawpop_ubs[best_index], models[best_index][1], eval_type=output_eval_type,
                                   features_col=features_col, is_bin=is_bin,
                                   return_modified_df=True, aggregate_for_skills=False))
    print('Train: ')
    print(predict_and_evaluate_dfs(models[best_index][0], train_list[best_index][0], train_list[best_index][1],
                                    rawpop_ubs[best_index], models[best_index][1], eval_type=output_eval_type,
                                            features_col=features_col, is_bin=is_bin,
                                                    return_modified_df=True, aggregate_for_skills=False))
    print('------------------------------------')
    return {i: (models[i], cs[i], qs[i], scores[i], rawpop_ubs[i], pre_norm_list[i], pcas_list[i])
            for i in range(len(models))}, best_index


def test_eval_for_wrapped_results(df_x_datapoints_with_ground_truth, df_y_all,
                                  results_dict, best_index, random_state, already_has_feature_selection=True,
                                  n_selection_features=None, test_proportion=0.3,
                                  based_on_year=False, also_based_on_skill=False, verbose=False, skills_to_ignore=None,
                                  skills_to_keep=None, skills_to_predict_zero=None,
                                  feature_col='tsfresh', baselines_to_test=None, pca_first=True,
                                  return_pred_df=False, binarisation_tools=None,
                                  is_binary=False, do_agg_eval=False, eval_type='prfs'):
    wrapped_results = results_dict[best_index]
    print(wrapped_results)
    best_model = wrapped_results[0]
    pca_model = wrapped_results[6]
    rawpop_upper_bound_dict = wrapped_results[4]
    prenormaliser = wrapped_results[5]

    class_balanced = (not based_on_year)
    print(random_state)
    print(class_balanced)
    print(best_index)


    truth_col = TRUTH_COL if is_binary else AUX_TRUTH_COL
    sampled_train_x_df, sampled_test_x_df, sampled_train_y_df, sampled_test_y_df = \
        create_train_test_split(df_x_datapoints_with_ground_truth, df_y_all,
                                test_proportion=test_proportion,
                                based_on_year=based_on_year,
                                also_based_on_skill=also_based_on_skill,
                                class_balanced=class_balanced,
                                random_state=random_state, truth_col=truth_col)


    if skills_to_ignore is not None:
        sampled_train_x_df = sampled_train_x_df.loc[
            sampled_train_x_df.Skill.apply(lambda x: x not in skills_to_ignore)]
        sampled_train_y_df = sampled_train_y_df.loc[
            sampled_train_y_df.Skills.apply(lambda x: x not in skills_to_ignore)]
        sampled_test_x_df = sampled_test_x_df.loc[
            sampled_test_x_df.Skill.apply(lambda x: x not in skills_to_ignore)]
        sampled_test_y_df = sampled_test_y_df.loc[
            sampled_test_y_df.Skills.apply(lambda x: x not in skills_to_ignore)]


    train_features = prenormaliser.transform(series_to_matrix(sampled_train_x_df[feature_col]))
    test_features = prenormaliser.transform(series_to_matrix(sampled_test_x_df[feature_col]))


    if not already_has_feature_selection and n_selection_features is not None:
        print('Oops, shouldn\'t be here')
        feature_reduction_model = SelectKBest(score_func=mutual_info_classif, k=n_selection_features)
        train_features = pca_model.fit_transform(train_features)
        test_features = pca_model.transform(test_features)

        train_features = feature_reduction_model.fit_transform(train_features,
                   sampled_train_y_df.loc[sampled_train_y_df.common_index.
                   apply(lambda x: x in sampled_train_x_df.common_index.values)].row_class.values)
        test_features = feature_reduction_model.transform(test_features)
    else:
        feature_reduction_model = pca_model[1]
        print(feature_reduction_model)
        pca_model = pca_model[0]
        if pca_first or feature_reduction_model is None:
            train_features = pca_model.transform(train_features)
            test_features = pca_model.transform(test_features)
            if feature_reduction_model is not None:
                train_features = feature_reduction_model.transform(train_features)
                test_features = feature_reduction_model.transform(test_features)
        else:
            train_features = feature_reduction_model.transform(train_features)
            test_features = feature_reduction_model.transform(test_features)
            train_features = pca_model.transform(train_features)
            test_features = pca_model.transform(test_features)

    sampled_train_x_df[feature_col] = train_features.tolist()
    sampled_test_x_df[feature_col] = test_features.tolist()

    results_to_return = dict()

    if binarisation_tools is not None:
        binary_gt = binarisation_tools[0]
        sampled_test_y_df = sampled_test_y_df.copy()
        sampled_train_y_df = sampled_train_y_df.copy()
        sampled_test_y_df['row_class'] = sampled_test_y_df.\
                            Skills.apply(lambda x: 1 if x in binary_gt[1] else 0)

        sampled_train_y_df['row_class'] = sampled_train_y_df.\
                            Skills.apply(lambda x: 1 if x in binary_gt[0] else 0)
        eligible_skills = binarisation_tools[1]
        rank_quantile = binarisation_tools[2]
        print('Yes binarisation')
        print(sampled_test_y_df.loc[sampled_test_y_df['row_class'] == 1].common_index.sum())
        print(sampled_test_y_df.loc[sampled_test_y_df['row_class'] == 1].Skills.values.tolist())
    else:
        print('No binarisation')
        print(sampled_test_y_df.loc[sampled_test_y_df['row_class'] == 1].common_index.sum())
        print(sampled_test_y_df.loc[sampled_test_y_df['row_class'] == 1].Skills.values.tolist())
        binary_gt = None
        eligible_skills = None
        rank_quantile = None

    print('Common Index Sum: ')
    print(sampled_test_y_df.common_index.sum())

    print('**********Test set, not aggregated')
    if rank_quantile is None:
        test_y_pred, test_y_eval, test_pred_positive_df, test_actual_positive_df, test_all_pred = \
            predict_and_evaluate_dfs(best_model[0], sampled_test_x_df, sampled_test_y_df,
                                 rawpop_upper_bounds=rawpop_upper_bound_dict, skills_to_keep=skills_to_keep,
                                 features_col=feature_col,
                                 normaliser=best_model[1], eval_type=eval_type, return_modified_df=True)
    else:
        test_y_pred, test_y_eval, test_pred_positive_df, test_actual_positive_df, test_all_pred = \
            predict_and_binarise_to_evaluate(best_model[0], sampled_test_x_df, sampled_test_y_df,
                                             eligible_skills[1], rank_quantile[1], skills_to_keep=skills_to_keep,
                                             features_col=feature_col, normaliser=best_model[1],
                                             eval_type=eval_type, return_modified_df=True)

    # print(''.join([str(int(x)) for x in test_y_pred]))

    if skills_to_predict_zero is not None:
        test_all_pred['pred'] = test_all_pred.apply(lambda x: x['pred']
                            if x['Skills'] not in skills_to_predict_zero else 0, axis=1)
        test_y_eval = evaluate_results(test_y_pred.pred.values, test_y_pred.row_class.values, 'prfs')

    results_to_return['test_unagg'] = test_y_eval
    if verbose:
        print(test_y_pred, test_y_eval, test_pred_positive_df, test_actual_positive_df)

    if verbose:
        print('-----------Baseline-----------')
    test_baseline_eval = \
        evaluate_results(np.array([1] * sampled_test_y_df.shape[0]), sampled_test_y_df[TRUTH_COL].values, eval_type='prfs')
    if verbose:
        print(test_baseline_eval)

    results_to_return['test_baseline_unagg'] = test_baseline_eval
    if do_agg_eval:
        if verbose:
            print('**********Test set, aggregated')

        test_y_pred_agg, test_y_eval, test_pred_positive_df, test_actual_positive_df, test_all_pred_agg = \
            predict_and_evaluate_dfs(best_model[0], sampled_test_x_df, sampled_test_y_df,
                                     rawpop_upper_bounds=rawpop_upper_bound_dict,
                                     features_col=feature_col,
                                     normaliser=best_model[1], eval_type=eval_type, return_modified_df=True,
                                     aggregate_for_skills=True)

        if verbose:
            print(test_y_pred_agg, test_y_eval, test_pred_positive_df, test_actual_positive_df)

        results_to_return['test_agg'] = test_y_eval

        if verbose:
            print('-----------Baseline-----------')
        agg_test_gt = sampled_test_y_df[['Skills', 'row_class']].groupby('Skills'). \
            apply(lambda x: 1 if x['row_class'].sum() > 0 else 0) \
            .values
        test_baseline_eval = \
            evaluate_results(np.array([1] * len(sampled_test_y_df.Skills.unique())), agg_test_gt
                             , eval_type=eval_type)
        if verbose:
            print(test_baseline_eval)

        results_to_return['test_baseline_agg'] = test_baseline_eval

    if verbose:
        print('**********Training set, not aggregated')

    if rank_quantile is None:
        train_y_pred, train_y_eval, train_pred_positive_df, train_actual_positive_df, train_all_pred = \
            predict_and_evaluate_dfs(best_model[0], sampled_train_x_df, sampled_train_y_df,
                                     rawpop_upper_bounds=rawpop_upper_bound_dict, skills_to_keep=skills_to_keep,
                                     normaliser=best_model[1], eval_type=eval_type, features_col=feature_col,
                                     return_modified_df=True)
    else:
        train_y_pred, train_y_eval, train_pred_positive_df, train_actual_positive_df, train_all_pred = \
            predict_and_binarise_to_evaluate(best_model[0], sampled_train_x_df, sampled_train_y_df,
                                             eligible_skills[0], rank_quantile[0], skills_to_keep=skills_to_keep,
                                             features_col=feature_col, normaliser=best_model[1],
                                             eval_type=eval_type, return_modified_df=True)


    results_to_return['train_unagg'] = train_y_eval

    if verbose:
        print(train_y_pred, train_y_eval, train_pred_positive_df, train_actual_positive_df)

    if verbose:
        print('-----------Baseline-----------')
    train_baseline_eval = \
        evaluate_results(np.array([1] * sampled_train_y_df.shape[0]), sampled_train_y_df[TRUTH_COL].values, eval_type='prfs')
    if verbose:
        print(train_baseline_eval
        )

    results_to_return['train_baseline_unagg'] = train_baseline_eval

    if do_agg_eval:
        if verbose:
            print('**********Training set, aggregated')

        train_y_pred, train_y_eval, train_pred_positive_df, train_actual_positive_df, train_all_pred = \
            predict_and_evaluate_dfs(best_model[0], sampled_train_x_df, sampled_train_y_df,
                                     rawpop_upper_bounds=rawpop_upper_bound_dict,
                                     normaliser=best_model[1], eval_type=eval_type, features_col=feature_col,
                                     return_modified_df=True, aggregate_for_skills=True)

        results_to_return['train_agg'] = train_y_eval

        if verbose:
            print(train_y_pred, train_y_eval, train_pred_positive_df, train_actual_positive_df)

        if verbose:
            print('-----------Baseline-----------')
        agg_test_gt = sampled_train_y_df[['Skills', 'row_class']].groupby('Skills'). \
            apply(lambda x: 1 if x['row_class'].sum() > 0 else 0) \
            .values
        train_baseline_eval = evaluate_results(np.array([1] * len(sampled_train_y_df.Skills.unique())), agg_test_gt
                                               , eval_type=eval_type)
        if verbose:
            print(train_baseline_eval)

        results_to_return['train_baseline_agg'] = train_baseline_eval

    if baselines_to_test is not None:
        for baseline_name in baselines_to_test:
            current_baseline_positives = list(baselines_to_test[baseline_name])
            current_test_y = sampled_test_y_df.copy()
            if '_' in current_baseline_positives[0] and '-' in current_baseline_positives[0]:
                current_test_y['pred'] = current_test_y.common_key.apply(lambda x:
                                                             1 if x in current_baseline_positives else 0)
            else:
                current_test_y['pred'] = current_test_y.Skills.apply(lambda x:
                                                             1 if x in current_baseline_positives else 0)
            if skills_to_keep is not None:
                current_test_y = current_test_y.loc[current_test_y.common_key.apply(lambda x: x in skills_to_keep)]
            results_to_return[baseline_name] = \
                evaluate_results(current_test_y.pred.values, current_test_y.row_class.values, eval_type=eval_type)
    if not return_pred_df:
        return results_to_return
    else:
        return results_to_return, test_all_pred


def interpret_model(clf_model, feature_names, feature_scores=None,
                    pca_model=None, n_features=None, dataframes=None, normaliser=None):
    if hasattr(clf_model, 'intercept_'):
        if feature_scores is None:
            if isinstance(clf_model.intercept_, list):
                scores = [-clf_model.intercept_[0]] + clf_model.coef_.flatten().tolist()
            else:
                scores = [-clf_model.intercept_] + clf_model.coef_.flatten().tolist()
        else:
            if isinstance(clf_model.intercept_, list):
                scores = [-clf_model.intercept_[0]] + feature_scores.flatten().tolist()
            else:
                scores = [-clf_model.intercept_] + feature_scores.flatten().tolist()
    else:
        if feature_scores is None:
            scores = clf_model.coef_.flatten().tolist()
        else:
            scores = feature_scores.flatten().tolist()
    if pca_model is None:
        names = feature_names
        if hasattr(clf_model, 'intercept_'):
            names = ['s_min']+names
        interpretation_df = pd.DataFrame({'Name': names,
                                          'Score': scores})
    else:
        names = list(range(len(clf_model.coef_.flatten())))
        if hasattr(clf_model, 'intercept_'):
            names = ['s_min'] + names
        og_names = feature_names
        features_list = [[(og_names[i], pca_model.components_[j,i])
                                        for i in (np.argsort(np.abs(pca_model.components_[j,:]).flatten()))[-20:]][::-1]
                                           for j in range(pca_model.components_.shape[0])]
        if hasattr(clf_model, 'intercept_'):
            features_list = ['None'] + features_list
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