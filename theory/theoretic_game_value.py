# imports
import numpy as np
from scipy.special import gammainc


########################################################################################################################
# Game value as a function of m,n,k
########################################################################################################################
def game_value(m, n, d, k):
    """"""
    if n > m:
        log_val = np.log((n * (m + k)) / (m * (n + k)))
        denominator = 2 * k * (n - m)
        x1 = (n * d * (m + k) * log_val) / denominator
        x2 = (m * d * (n + k) * log_val) / denominator
        v = 0.5 + 0.5 * (gammainc(d/2, x1) - gammainc(d/2, x2))
    else:
        v = 0.5
    return v


def game_value_as_func_of_n(m, n_max, d, k):
    """"""
    v = np.zeros((n_max,))
    n_array = np.arange(1, n_max + 1)
    for n in n_array:
            v[n - 1] = game_value(m, n, d, k)
    return n_array, v


########################################################################################################################
# Game value as a function of delta, rho
########################################################################################################################