# imports
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)
from theory.theoretic_game_value import game_value_rho_delta


########################################################################################################################
# Functions
########################################################################################################################
def plot_game_value_of_inv_delta_for_diff_rho(game_value_func, inv_delta_max, d, rho_list, linewidth=2.):
    """"""
    for rho in rho_list:
        v = np.zeros((inv_delta_max,))
        inv_delta_array = np.arange(1, inv_delta_max + 1)
        for i in range(inv_delta_array.shape[0]):
            inv_delta = inv_delta_array[i]
            delta = 1. / inv_delta
            v[i]  = game_value_func(d=d, rho=rho, delta=delta)
        plt.plot(inv_delta_array, v, label=r'$\rho$ = {}'.format(rho), linewidth=linewidth)
    plt.xlabel('n/m')
    plt.ylabel('Game Value')
    plt.title('d = {}'.format(d))
    plt.legend(loc='lower right')
    plt.grid(color='k', alpha=0.2, axis='both', which='both')
    plt.show()


def main(args):
    """"""
    plot_game_value_of_inv_delta_for_diff_rho(
        game_value_func=game_value_rho_delta, inv_delta_max=args.max_n_over_m, d=args.d, rho_list=args.rho_list
    )


def get_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=100, help='The dimension of observations')
    parser.add_argument('--max_n_over_m', type=int, default=100, help='Max value of x-axis in plot')
    parser.add_argument('--rho_list', type=int, nargs='+', default=(0.1, 1, 2, 5, 10),
                        help='List of rho values for a which the game value will be plotted')
    return parser.parse_args()


########################################################################################################################
# App
########################################################################################################################
if __name__ == '__main__':
    mpl.rcParams['font.size'] = 16.
    main(get_args())