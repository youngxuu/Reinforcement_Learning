#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/6 14:59
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : Monte_Carlo_Control.py
# @Software: PyCharm

import numpy as np
from RLBrain_LookupTable.utils import state_action_index, state_index
from RLBrain_LookupTable.Base_Brain import BaseBrain


class MonteCarloControl(BaseBrain):
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
        super().__init__(action_all, state_all, gamma)
        self.n_0 = n_0
        self._state_action_num = np.zeros(self._state_action_dim, dtype='int32')
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
        from Environment.EER_experiment import Environment

        env = Environment()
        state_all = [list(range(0, 200))]
        action_all = env.act_space
        rl_brain = MonteCarloControl(action_all, state_all)
        for i in range(500):
            done = False
            s = env.reset()
            while not done:
                act = rl_brain.choose_action(s)
                s_, r, done = env.step(act)
                rl_brain.store_transactions(s, act, r)
                s = s_
            rl_brain.learn()
        print(rl_brain._state_action_value)
        print(np.argmax(rl_brain._state_action_value, axis=0) + 1)




