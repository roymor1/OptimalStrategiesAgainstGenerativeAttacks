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
from theory.theoretic_game_value import game_value_rho_delta, ml_attacker_game_value_rho_delta, game_value_diff_ml_vs_opt_rho_delta

########################################################################################################################
# Constants
########################################################################################################################
EPS = 1e-6
GAME_VALUE_FUNCTIONS = {
    'nash_game_value': game_value_rho_delta,
    'ml_attacker_game_value': ml_attacker_game_value_rho_delta,
    'game_value_diff_ml_vs_opt': game_value_diff_ml_vs_opt_rho_delta
}

########################################################################################################################
# Functions
########################################################################################################################
def plot_game_value_of_rho_delta(game_value_func, d, rho_log_range=(-4, 4), delta_range=(EPS,1), value_range=(0.5, 1.)):
    """"""
    # make these smaller to increase the resolution
    n_points = 1000

    # generate 2 2d grids for the x & y bounds
    rho, delta = np.meshgrid(
        np.logspace(rho_log_range[0], rho_log_range[1], num=n_points, endpoint=True),
        np.linspace(delta_range[0], delta_range[1], num=n_points, endpoint=True)
    )
    v = np.zeros_like(delta)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            v[i,j] = game_value_func(d=d, rho=rho[i,j], delta=delta[i,j])
    v = v[:-1, :-1]
    plt.pcolor(rho, delta, v, vmin=value_range[0], vmax=value_range[1])
    plt.colorbar()
    plt.xscale('log')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\delta$')
    plt.title('d = {}'.format(d))
    plt.show()


def main(args):
    """"""
    plot_game_value_of_rho_delta(
        game_value_func=GAME_VALUE_FUNCTIONS[args.plot_type],
        d=args.d,
        value_range=(0., 0.5) if (args.plot_type == 'game_value_diff_ml_vs_opt') else (0.5, 1.)
    )


def get_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=100, help='The dimension of observations')
    parser.add_argument('--plot_type',
                        default='nash_game_value',
                        help='The type of plot:\n'
                             '"nash_game_value": Plot the game value as a function of delta and rho.\n'
                             '"ml_attacker_game_value": '
                             'Plot the game value as a function of delta and rho when fixing the attacker to be the sub-optimal ml attacker.\n'
                             '"game_value_diff_ml_vs_opt": '
                             'Plot the difference in game value achieved by the ml attacker and the optimal attacker, '
                             'as a function of delta and rho.')
    return parser.parse_args()


########################################################################################################################
# App
########################################################################################################################
if __name__ == '__main__':
    mpl.rcParams['font.size'] = 16.
    main(get_args())