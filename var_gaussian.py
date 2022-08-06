import pandas as pd
import numpy as np
import scipy.stats as ss

def value_at_rik_gaussian(return_series, level = 5, modified = False):
    """
    Calculate the Parametric Gaussian value at risk (VaR) of a Series or DataFrame:
    1- If "modified" is False, Gaussian method calculates the VaR.
    2- If "modified" is True, then the modified VaR is returned using the Cornish-Fisher modification.
    3- Let's define skewness function alternative to scipy.stats.skew()
    4- Let's define kurtosis function alternative to scipy.stats.kurtosis()
    """
    ############################## Gaussian ################################

    z = ss.norm.ppf(level / 100)

    ############ Skewness and Kurtosis for Cornish-Fisher ##################

    def skewness(return_series):
        demeaned_return = return_series - return_series.mean()
        sigma_return = return_series.std(ddof = 0)
        exp = (demeaned_return ** 3).mean()
        return exp / sigma_return ** 3
    
    def kurtosis(return_series):
        demeaned_return = return_series - return_series.mean()
        sigma_return = return_series.std(ddof = 0)
        exp = (demeaned_return ** 4).mean()
        return exp / sigma_return ** 4

    ############################ Cornish-Fisher ##############################
    
    if modified:
        s = skewness(return_series)
        k = kurtosis(return_series)
        z = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * (k - 3) / 24 - (2 * z ** 3 - 5 * z) * (s ** 2) / 36)
    return -(return_series.mean() + z * return_series.std(ddof=0))
