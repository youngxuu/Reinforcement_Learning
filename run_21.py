#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/7 21:52
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : run_21.py
# @Software: PyCharm

__version__ = "1.0"
from RLBrain_LookupTable.Monte_Carlo_Control import MonteCarloControl
from Environment.easy_21 import Easy21
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def main():
    env = Easy21()
    action_all = ['hit', 'stick']
    state_all = [list(range(1, 22)), list(range(1, 11))]
    rl_brain = MonteCarloControl(action_all, state_all)
    for i in range(100000):
        # print('game start %d' % (i + 1))
        # print('\n')
        done = False
        s = env.reset()
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


def main_():

    from RLBrain_DeepRL.DeepQLearning import DeepQLearning
    env = Easy21()
    action_all = ['hit', 'stick']
    # state_all = [list(range(1, 22)), list(range(1, 11))]
    # rl brain
    rl_brain = DeepQLearning(n_features=2, n_actions=2, actions=action_all,
                             learning_rate=0.01, e_greedy=0.1,
                             reward_decay=0.9, output_graph=True, double=True, EnQ=True)
    t0 = time.time()
    for i in range(10000):
        # print('game start %d' % (i + 1))
        # print('\n')
        done = False
        s = env.reset()
        s = np.array(s, dtype='float32')
        while not done:
            player_act = rl_brain.choose_action(s)
            # print(player_act)
            s1, r, done = env.step(player_act)
            s1 = np.array(s1, dtype='float32')
            rl_brain.store_transactions(s, player_act, s1, r, done)
            # print(s)
            s = s1
        rl_brain.learn()
        if i % 100 == 0:
            t1 = time.time()
            print('epoch: %d;    learning iter: %d;    time cost: %.2f' % (i, rl_brain.learn_iter, t1 - t0))
            t0 = t1
    state_all = []
    for p in range(1, 22):
        for d in range(1, 11):
            state_all.append([p, d])
    q_pred = rl_brain.predict(np.array(state_all, dtype='float32'))
    print(q_pred)
    q_values = np.max(q_pred, axis=1).reshape(21, 10)
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(1, 11, 1)
    y = np.arange(1, 22, 1)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, q_values, rstride=1, cstride=1, cmap='rainbow')
    plt.show()


if __name__ == '__main__':
    main_()
