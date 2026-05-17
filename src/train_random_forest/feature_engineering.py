""" This file contains the implementation of the feature engineering steps that we apply to the data before feeding it to the random forest model. You can find the function `delta_date_feature` that computes a feature based on the difference in days between a date and the most recent date in its column. You can add more functions to this file if you want to create more features.
"""

import pandas as pd
import numpy as np


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()
