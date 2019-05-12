#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
import time
import numpy as np
import matplotlib.pyplot as plt

from Environment.day_to_day_trans_flow import Environment
from RLBrain_DeepRL.DeepQLearning import DeepQLearning


def plot_link_flow(link, actions):
    """
    plot the daily link flow and total cost per day w.r.t actions
    :param link: which link to charge
    :param actions: how much fees to charge
    :return:
    """
    steps = len(actions)
    tc_per_day = np.zeros(steps, dtype='float32')  # store reward
    path_no_toll = np.zeros((steps, 12), dtype='float32')  # store daily link flow
    env = Environment()
    s = env.reset()
    for i, act in enumerate(actions):
        path_no_toll[i, :] = s
        one_hot = np.zeros(12, dtype='float32')
        one_hot[link - 1] = 1
        action = act * one_hot
        r, s_, done = env.step(action=action)
        s = s_
        tc_per_day[i] = r

    # plot the link flow changes w.r.t actions
    plt.plot(path_no_toll[:, 0] / path_no_toll[0, 0], color='green', label='link1')
    plt.plot(path_no_toll[:, 1] / path_no_toll[0, 1], color='red', label='link2')
    plt.plot(path_no_toll[:, 2] / path_no_toll[0, 2], color='blue', label='link3')
    # plt.plot(path_no_toll[:, 3]/path_no_toll[0, 3], color='orange', label='link4')
    plt.plot(path_no_toll[:, 9] / path_no_toll[0, 9], color='yellow', label='link10')
    plt.plot(path_no_toll[:, 11] / path_no_toll[0, 11], color='skyblue', label='link12')
    plt.legend()
    plt.xlim(0, steps)
    plt.ylim(0.2, 2)
    plt.xlabel('day')
    plt.ylabel('normalized link flow')
    plt.show()

    # plot the daily total cost changes w.r.t actions
    plt.plot(tc_per_day, color='green', label='total cost')
    plt.legend()
    plt.xlim(0, steps)
    plt.xlabel('day')
    plt.ylabel('total cost / day')
    plt.show()


def main():
    venv = Environment()
    # all possible actions (possible fees to charge)
    actions = (1. * np.arange(0, 6, dtype='float32')).tolist()
    # number of possibles actions
    n_actions = len(actions)
    # rl brain
    rl_brain = DeepQLearning(n_features=12, n_actions=n_actions, actions=actions,
                             learning_rate=0.01, e_greedy=0.1,
                             reward_decay=0.9, output_graph=True, double=True, EnQ=True)
    # specify link to set tool booth
    charged_link = 1
    # number of learning epochs
    epochs = 100
    # store the total reward per 10 epochs
    store_total_reward = []
    stored_epochs = []
    # store the learned policy at the last epoch
    policy = []
    start_t = time.time()
    for epoch in range(epochs):
        s = venv.reset()
        total_r = 0
        delta_flows = []
        for t in range(30):
            a = rl_brain.choose_action(s)
            # action transform
            one_hot = np.zeros(12, dtype='float32')
            one_hot[charged_link - 1] = 1
            action = a * one_hot
            r, s_, done = venv.step(action)
            delta_flow = np.sum(np.abs(s_ - s))
            delta_flows.append(delta_flow)
            if t >= 2 and np.sum(delta_flows[-3:]) <= 1.:
                r_ = r + 100000
            else:
                r_ = r
            rl_brain.store_transactions(s, a, s_, r_, done)
            s = s_
            total_r += r
            if epoch == epochs - 1:
                policy.append(a)
            if done:
                break
            rl_brain.learn()

        if epoch % 5 == 0:
            stored_epochs.append(epoch)
            store_total_reward.append(total_r)
            end_time = time.time()
            print('epoch: %d, '
                  'total time cost per 10 epoch: %.2f ;'
                  'total reward: %0.2f; learning iter: %d'
                  % (epoch, end_time - start_t, total_r, rl_brain.learn_iter))
            start_t = end_time
    # plot the daily total cost changes w.r.t epochs
    plt.plot(store_total_reward, color='green', label='total cost')
    plt.legend()
    # plt.xticks(np.arange(len(stored_epochs)).tolist(), stored_epochs)
    plt.xlabel('epochs')
    plt.ylabel('total cost / epoch')
    plt.show()
    # plot the daily actions changes
    plt.plot(policy, color='green', label='fees')
    plt.legend()
    # plt.xticks(np.arange(len(stored_epochs)).tolist(), stored_epochs)
    plt.xlabel('days')
    plt.ylabel('fees / day')
    plt.show()
    plot_link_flow(charged_link, policy)


if __name__ == '__main__':
    main()



