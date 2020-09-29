import numpy as np

FEATURE_COL='Features'

TIME_PERIODS = {
    '2017-2018': (np.datetime64('2017-01-01'), np.datetime64('2018-01-01')),
    '2018-2019': (np.datetime64('2018-01-01'), np.datetime64('2019-01-01')),
    '2019-2020': (np.datetime64('2019-01-01'), np.datetime64('2020-01-01')),
    '2018-2020': (np.datetime64('2018-01-01'), np.datetime64('2020-01-01')),
    '2017-2019': (np.datetime64('2017-01-01'), np.datetime64('2019-01-01')),
    '2017-2020': (np.datetime64('2017-01-01'), np.datetime64('2020-01-01'))
}

FEATURES_TO_COMPUTE=['linreg', 'tsfresh']

CV_FOLDS=5
C_LIST = np.logspace(start=-5, stop=4, num=10, base=10)
QUANTILES = np.linspace(0.7, 0.95, num=6)