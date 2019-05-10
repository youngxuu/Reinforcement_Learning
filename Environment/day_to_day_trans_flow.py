#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
import numpy as np
from scipy.optimize import minimize


class Environment(object):
    """
    Final Project
    Model-free control of day-to-day dynamics with DQN
    see ref:
    """
    def __init__(self, lamb=0.7, alpha=0.7, t_a=1500):
        """
        the step-size  and the weight parameter .
        :param lamb: the step-size
        :param alpha: the weight parameter
        :param t_a: free flow time
        """
        self.lamb = lamb
        self.alpha = alpha
        self.t_a = t_a
        # node link flow constraint see ref doc
        self.A = np.array(
            [[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 1, -1, 0, -1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 1],
             [0, 0, 0, 0, -1, 0, -1, 0, 0, 1, 0, 0]], dtype='float32')
        self.b = np.array([2000, 0, 0, 0, 0, 0, 0, 0], dtype='float32')

        self.state = None
        #  The testing scenario is that a 50% capacity reduction on Link 1 takes place at day 0.
        self.C_a_changed = np.array(
            [500, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
            dtype='float32')

    def reset(self):
        """
        reset the environment, ie. set the state to the path link flow of day 0
        :return: init state
        """
        init_state = np.array(
                            [1000, 500, 1000, 500, 500, 500, 500, 500, 500, 1000, 500, 1000],
                            dtype='float32')
        self.state = init_state.copy()
        return init_state

    def step(self, action):
        """
        specify a link to build a tool booth and take dynamic fees
        :param action: numpy.array of shape (12,) ,
                    specify with link to build the tool booth
                    and how much fees charged.
        :return: -T_c, state_, done; the cost of the network, the path link flow of each link
         and whether the system becomes stable after we take a specific action
        """
        done = False
        # trans cash value to time value
        action = 360 * action
        # solve the system to get the daily changes in path link flow
        # constraints of the system
        b_eq = 1 / self.alpha * (self.b + (self.alpha - 1) * np.dot(self.A, self.state))
        res = minimize(fun=lambda y: np.sum((1 - self.lamb) * y ** 5 / self.C_a_changed ** 4
                                            + ((100 / 3) * self.lamb * (1 + action / self.t_a)
                                            + (10 * self.lamb - 5) * self.state ** 4 / self.C_a_changed ** 4) * y
                                            + 4 * (1 - self.lamb) * self.state ** 5 / self.C_a_changed ** 4),
                       x0=np.random.uniform(0, 1, 12),
                       bounds=tuple((tuple((0, None)) for vec in range(12))),
                       constraints=({'type': 'eq', 'fun': lambda y: np.dot(self.A, y) - b_eq}))
        # changes in the network (path link flow)
        delta_x = self.alpha * (res.x - self.state)
        # update state
        self.state += delta_x
        state_ = self.state.copy()
        # calculate the reward(negative cost)
        cost = np.sum(self.t_a + action
                      + 0.15 * self.t_a * self.state ** 4 / self.C_a_changed ** 4)
        # whether the network goes to stable, if stable, add a negative cost to the reward
        if np.sum(np.abs(delta_x)) < 10:
            done = True
            cost += -100000
        return -cost, state_, done


if __name__ == '__main__':
    path_no_toll = np.zeros((30, 12), dtype='float32')  # store daily link flow
    tc_per_day = np.zeros(30, dtype='float32')  # store reward
    t = 0
    env = Environment()
    s = env.reset()
    j = 1
    for i in range(30):
        path_no_toll[i, :] = s
        one_hot = np.zeros(12, dtype='float32')
        one_hot[j-1] = 1
        action = 0. * one_hot
        r, s_, done = env.step(action=action)
        s = s_
        tc_per_day[i] = r
    print(np.sum(tc_per_day))

    import matplotlib.pyplot as plt
    plt.plot(path_no_toll[:, 0] / path_no_toll[0, 0], color='green', label='link1')
    plt.plot(path_no_toll[:, 1] / path_no_toll[0, 1], color='red', label='link2')
    plt.plot(path_no_toll[:, 2] / path_no_toll[0, 2], color='blue', label='link3')
    # plt.plot(path_no_toll[:, 3]/path_no_toll[0, 3], color='orange', label='link4')
    plt.plot(path_no_toll[:, 9] / path_no_toll[0, 9], color='yellow', label='link10')
    plt.plot(path_no_toll[:, 11] / path_no_toll[0, 11], color='skyblue', label='link12')
    plt.legend()
    plt.xlim(0, 30)
    plt.ylim(0.2, 2)
    plt.xlabel('day')
    plt.ylabel('normalized linkflow')
    plt.show()