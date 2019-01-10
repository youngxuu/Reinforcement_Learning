#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/6 14:59
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : Monte_Carlo_Control.py
# @Software: PyCharm

import numpy as np


def state_index(state, state_all):
    """
    transform state to index
    this function is used for transform state value into numpy index,

    :param state: list current state needed to transform
           state_all:a list of all possible states for every dimensions
                     example [['1'], [1, 2, 3]]
    :return: tuple of index
    """
    state_dir = []
    for i in range(len(state_all)):
        dim = i
        state_ = {}
        for j, state_j in enumerate(state_all[dim]):
            state_.update({state_j: j})
        state_dir.append(state_)

    return tuple([state_dir[i][state_i] for i, state_i in enumerate(state)])


def state_action_index(action, state, action_dict, state_all):
    """
    transform action state to index
    this function is used for transform action state value into numpy index,
    :param action: action
    :param state: list, current state needed to transform
    :param action_dict: dict, a dictionary with keys(action) and values(index) all possible action an agent can take
    :param state_all: a list of all possible states for every dimensions
                     example [['1'], [1, 2, 3]]
    :return: tuple of index
    """
    # print(action_dict)
    state_act_dir = []
    state_act_dir.append(action_dict)
    for i in range(len(state_all)):
        dim = i
        state_ = {}
        for j, state_j in enumerate(state_all[dim]):
            state_.update({state_j: j})
        state_act_dir.append(state_)

    state_act = [action]
    state_act += state
    for i, state_i in enumerate(state_act):
        # print('state_action_index debug', i, state_i)
        slice_ = state_act_dir[i]
        # print(slice_)

    return tuple([state_act_dir[i][state_i] for i, state_i in enumerate(state_act)])


class MonteCarloControl(object):
    """
    Monte Carlo Control for reinforcement learning
    Parameters
    ----------
    action_all: list, all possible actions an agent have
    state_all: list, all possible state the RL environment have(look_up table)
    n_0: int scalar, hyper-parameter of e greedy exploration

    gamma: float, 0 < gamma < 1, reward discount rate
    """
    def __init__(self, action_all, state_all, n_0=100, gamma=0.01):
        self.act_num = len(action_all)
        act_dict = {}
        act_dict.update({act: i for i, act in enumerate(action_all)})
        self.act_dict = act_dict
        self.idx_act = {v: k for k, v in self.act_dict.items()}
        self.state_all = state_all
        state_dim = []
        for i in range(len(self.state_all)):
            state_dim.append(len(self.state_all[i]))
        self.state_dim = state_dim
        self.n_0 = n_0
        if not isinstance(gamma, float):
            raise TypeError("Threshold is not a float value!")
        if not 0 < gamma < 1:
            raise ValueError("Threshold must >= 0 and <= 1!")
        self.gamma = gamma
        _state_action_dim = [self.act_num] + self.state_dim
        # print(_state_action_dim)
        self._state_action_num = np.zeros(_state_action_dim, dtype='int32')
        self._state_action_value = np.zeros(_state_action_dim, dtype='float32')
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def choose_action(self, state):
        """
        choose an action based on the current state according to the e greedy exploration
        parameters
        ----------
        state: list ,
                current state of the agent(please note that:monte carlo updates the state
                action value by a look up table, thus the state is a discrete space).

        """
        if not isinstance(state, list):
            raise TypeError('"state" must be list type')
        state_idx = state_index(state, self.state_all)

        state_num_all = np.sum(self._state_action_num, axis=0)
        # print(state_num_all)
        state_num = state_num_all[state_idx]
        epsilon = self.n_0 / (self.n_0 + state_num)
        if np.random.uniform(0, 1) > epsilon:
            act_idx = np.argmax(self._state_action_value, axis=0)[state_idx]
        else:
            act_idx = np.random.randint(self.act_num)
        return self.idx_act[act_idx]

    def store_transactions(self, state, action, reward):
        """
        store state, action and reward in an episode.
        :param state: list of state
        :param action: action
        :param reward: reward, float.
        :return: None
        """
        self.ep_obs.append(state)
        self.ep_as.append(action)
        self.ep_rs.append(reward)

    def learn(self):
        """
        learning the state action values by MonteCarlo,
        update the action value using data collected by an episode.
        :return: None
        """
        # discount the reward and calculate Gt
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype='float32')
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add*self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        for i in range(len(self.ep_obs)):
            state_act_inx = state_action_index(self.ep_as[i], self.ep_obs[i], self.act_dict, self.state_all)
            self._state_action_num[state_act_inx] += 1
            self._state_action_value[state_act_inx] += (1/self._state_action_num[state_act_inx]) * \
                                                       (discounted_ep_rs[i] - self._state_action_value[state_act_inx])
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []


if __name__ == '__main__':
    mc = MonteCarloControl(['hit', 'stick'], [[1, 2], [1, 3]])
    act = mc.choose_action([1, 3])
    print(act)

    mc.store_transactions([1, 3], act, 1)
    mc.learn()




