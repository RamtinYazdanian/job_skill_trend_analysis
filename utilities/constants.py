import numpy as np

SMOOTH_N = 3
SMOOTH_ALPHA = 0.4

LINREG_FEATURES = ['Velocity', 'Acceleration', 'Spikiness', 'Intercept']

QUESTION_NAMES = {0: 'Main', 1: 'YesCol', 2: 'NoCol'}

UNSURE = 'Unsure'
YES = 'Yes'
NO = 'No'
BEFORE17 = 'Earlier than 2017'

TIME_PERIODS = {
    '2017-2018': (np.datetime64('2017-01-01'), np.datetime64('2018-01-01')),
    '2018-2019': (np.datetime64('2018-01-01'), np.datetime64('2019-01-01')),
    '2019-2020': (np.datetime64('2019-01-01'), np.datetime64('2020-01-01')),
    '2018-2020': (np.datetime64('2018-01-01'), np.datetime64('2020-01-01')),
    '2017-2019': (np.datetime64('2017-01-01'), np.datetime64('2019-01-01')),
    '2017-2020': (np.datetime64('2017-01-01'), np.datetime64('2020-01-01'))
}