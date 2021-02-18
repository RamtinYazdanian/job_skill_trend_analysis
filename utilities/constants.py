SMOOTH_N = 3
SMOOTH_ALPHA = 0.4

FEATURE_NAMES = {'linreg': ['Velocity', 'Acceleration', 'Spikiness', 'Intercept'],
                 'linreg_nointercept': ['Velocity', 'Acceleration', 'Spikiness'],
                 }

BASIC_FEATURES = {
    'mean', 'median', 'variance', 'minimum', 'maximum', 'length', 'abs_energy', 'absolute_sum_of_changes',
    'first_location_of_maximum', 'first_location_of_minimum', 'has_duplicate', 'has_duplicate_max',
    'has_duplicate_min', 'last_location_of_maximum', 'last_location_of_minimum', 'mean_abs_change',
    'mean_change', 'standard_deviation'
}

BASIC_PLUS_FEATURES = {
    'skewness', 'kurtosis', 'mean', 'median', 'variance', 'minimum',
    'maximum', 'length', 'abs_energy', 'absolute_sum_of_changes',
    'first_location_of_maximum', 'first_location_of_minimum', 'has_duplicate', 'has_duplicate_max',
    'has_duplicate_min', 'last_location_of_maximum', 'last_location_of_minimum', 'mean_abs_change',
    'mean_change', 'standard_deviation'
}

BASIC_PLUSPLUS_FEATURES = {
    'mean', 'median', 'variance', 'minimum', 'maximum', 'standard_deviation', 'abs_energy',
    'variance_larger_than_standard_deviation', 'absolute_sum_of_changes', 'longest_strike_above_mean',
    'longest_strike_below_mean'
}

BASIC_TRIPLEPLUS_FEATURES = {
    'mean', 'median', 'variance', 'minimum', 'maximum', 'standard_deviation', 'abs_energy',
    'absolute_sum_of_changes', 'longest_strike_above_mean', 'longest_strike_below_mean'
}



SUPER_BASIC_FEATURES = {
    'mean', 'median', 'variance', 'minimum', 'maximum', 'standard_deviation', 'abs_energy'
}

QUESTION_NAMES = {0: 'Main', 1: 'YesCol', 2: 'NoCol'}

UNSURE = 'Unsure'
YES = 'Yes'
NO = 'No'
BEFORE17 = 'Earlier than 2017'

POP_COL='Job Postings Raw'
TRUTH_COL='row_class'

X_REFCOL = 'Skill'
Y_REFCOL = 'Skills'