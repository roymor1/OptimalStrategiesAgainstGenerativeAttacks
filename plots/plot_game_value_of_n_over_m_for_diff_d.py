# imports
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)
from theory.theoretic_game_value import game_value_as_func_of_n

########################################################################################################################
# Functions
########################################################################################################################
def plot_game_value_of_n_div_m_for_diff_d(m, n_max, d_list, k, linewidth):
    """"""
    for d in d_list:
        n_array, v  = game_value_as_func_of_n(m=m, n_max=n_max, d=d, k=k)
        plt.plot(n_array, v, label='d = {}'.format(d), linewidth=linewidth)
    plt.xlabel('n/m')
    plt.ylabel('Game Value')
    # plt.xticks(np.arange(0, n_max + 1, 5))
    plt.legend(loc='lower right')
    plt.grid(color='k', alpha=0.2, axis='both', which='both')
    plt.show()


########################################################################################################################
# App
########################################################################################################################
if __name__ == '__main__':
    mpl.rcParams['font.size'] = 16.
    plot_game_value_of_n_div_m_for_diff_d(m=1, n_max=100, d_list=[1, 2, 5, 10, 20, 100], k=10, linewidth=2.)