#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/7 21:52
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : run_MC.py
# @Software: PyCharm

__version__ = "1.0"
from RLBrain_LookupTable.Monte_Carlo_Control import MonteCarloControl
from Environment.easy_21 import Easy_21
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def main():
    env = Easy_21()
    action_all = ['hit', 'stick']
    state_all = [list(range(1, 22)), list(range(1, 11))]
    rl_brain = MonteCarloControl(action_all, state_all)
    for i in range(100000):
        # print('game start %d' % (i + 1))
        # print('\n')
        done = False
        s = env.restart()
        while not done:
            player_act = rl_brain.choose_action(s)
            # print(player_act)
            s1, r, done = env.step(player_act)
            rl_brain.store_transactions(s, player_act, r)
            # print(s)
            s = s1
        rl_brain.learn()
    # print(rl_brain._state_action_value)
    print(np.argmax(rl_brain._state_action_value, axis=0))

    q_values = np.max(rl_brain._state_action_value, axis=0)
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(1, 11, 1)
    y = np.arange(1, 22, 1)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, q_values, rstride=1, cstride=1, cmap='rainbow')
    plt.show()


if __name__ == '__main__':
    main()
