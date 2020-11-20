import numpy as np

FEATURE_COL='linreg_nointercept'

TIME_PERIODS = {
    '2017-2018': (np.datetime64('2017-01-01'), np.datetime64('2018-01-01')),
    '2018-2019': (np.datetime64('2018-01-01'), np.datetime64('2019-01-01')),
    '2019-2020': (np.datetime64('2019-01-01'), np.datetime64('2020-01-01')),
    '2018-2020': (np.datetime64('2018-01-01'), np.datetime64('2020-01-01')),
    '2017-2019': (np.datetime64('2017-01-01'), np.datetime64('2019-01-01')),
    '2017-2020': (np.datetime64('2017-01-01'), np.datetime64('2020-01-01'))
}

FEATURES_TO_COMPUTE=['linreg', 'linreg_nointercept', 'tsfresh']

TS_N_FEATURES = 10

CV_FOLDS=5
C_LIST = np.logspace(start=-5, stop=3, num=9, base=10)
DECISION_TREE_PARAM_LIST = [
    {'max_depth': x, 'min_samples_split': y, 'max_features': z}
    for x in range(3,10) for y in range(2,4) for z in ['auto']
]
QUANTILES = np.linspace(0.85, 1, num=4)