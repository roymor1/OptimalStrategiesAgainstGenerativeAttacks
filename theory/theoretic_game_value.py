# imports
import argparse
import numpy as np
from scipy.special import gammainc


########################################################################################################################
# Game value as a function of m,n,k
########################################################################################################################
def game_value_mnk(m, n, d, k):
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
            v[n - 1] = game_value_mnk(m, n, d, k)
    return n_array, v


########################################################################################################################
# Game value as a function of delta, rho
########################################################################################################################
def game_value_rho_delta(d, rho, delta):
    """"""
    if delta < 1:
        log_val = np.log((1. + rho) / (delta + rho))
        denominator = 2 * (1 - delta)
        x1 = d * (1 + rho) * log_val / denominator
        x2 = d * (delta + rho) * log_val / denominator
        v = 0.5 + 0.5 * (gammainc(d/2, x1) - gammainc(d/2, x2))
    else:
        v = 0.5
    return v


def ml_attacker_game_value_rho_delta(d, rho, delta):
    """"""
    log_val = np.log((1. + rho + delta) / (delta + rho))
    denominator = 2.
    x1 = d * (1 + rho + delta) * log_val / denominator
    x2 = d * (delta + rho) * log_val / denominator
    v = 0.5 + 0.5 * (gammainc(d/2, x1) - gammainc(d/2, x2))
    return v


def game_value_diff_ml_vs_opt_rho_delta(d, rho, delta):
    return ml_attacker_game_value_rho_delta(d, rho, delta) - game_value_rho_delta(d, rho, delta)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=1, help='m: the number of leaked observations')
    parser.add_argument('-n', type=int, default=5, help='n: the number of test observations')
    parser.add_argument('-k', type=int, default=10, help='k: the number of registration observations')
    parser.add_argument('-d', type=int, default=100, help='d: the dimension of observations')
    return parser.parse_args()

########################################################################################################################
# Unit Test
########################################################################################################################
if __name__ == '__main__':
    args = get_args()
    v = game_value_mnk(m=args.m, n=args.n, k=args.k, d=args.d)
    print(v)