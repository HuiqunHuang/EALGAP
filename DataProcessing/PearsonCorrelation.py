# coding=utf-8
import datetime
import numpy as np
import os
import pandas as pd
import math
from scipy import stats

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
def Pearson_Correlation_Coefficient(x1_list, x2_list):
    # r: (float) Pearson’s correlation coefficient
    # p: (float) Two-tailed p-value
    # Correlation coefficients between - 1 and +1 with 0 implying no correlation.
    # Correlations of -1 or +1 imply an exact linear relationship.
    # Positive correlations imply that as x increases, so does y.
    # Negative correlations imply that as x increases, y decreases

    # The p-value roughly indicates the probability of an uncorrelated system producing datasets that
    # have a Pearson correlation at least as extreme as the one computed from these datasets.
    r, p = stats.pearsonr(x1_list, x2_list)
    return r, p

