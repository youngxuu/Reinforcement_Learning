#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/29 11:06
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : SARSA.py
# @Software: PyCharm

import numpy as np
from RLBrain_LookupTable.utils import state_index, state_action_index
from RLBrain_LookupTable.Base_Brain import BaseBrain


class Sarsa(BaseBrain):
    """
    an implementation of sarsa lambda.
        Parameters
    ----------
    action_all: list, all possible actions an agent have
    state_all: list, all possible state the RL environment have(look_up table)
    lamb: float, 0 < lamb < 1, sarsa lambda
    gamma: float, 0 < gamma < 1, reward discount rate
    epsilon:float, 0  < epsilon < 1, epsilon greedy rate
    alpha: float, 0  < alpha < 1, learning rate
    """
    def __init__(self, action_all, state_all, gamma=0.01, lamb=0.1, epsilon=0.1, alpha=0.1):
        super().__init__(action_all, state_all, gamma, epsilon)
        self.E = np.zeros(self._state_action_dim, dtype='float32')
        self.lamb = lamb
        self.alpha = alpha

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
        if np.random.uniform(0, 1) > self.epsilon:
            act_idx = np.argmax(self._state_action_value, axis=0)[state_idx]
        else:
            act_idx = np.random.randint(self.act_num)
        return self.idx_act[act_idx]

    def learn(self, state, action, reward, state_, action_, done):

        state_act_index = state_action_index(action, state, self.act_dict, self.state_all)
        self.E[state_act_index] += 1
        if done:
            delta = reward - self._state_action_value[state_act_index]
        else:
            state_act_index_ = state_action_index(action_, state_, self.act_dict, self.state_all)
            delta = reward + self.gamma * self._state_action_value[state_act_index_] - \
                    self._state_action_value[state_act_index]
        self._state_action_value += self.alpha * delta * self.E
        self.E = self.gamma*self.lamb*self.E


if __name__ == '__main__':
    from Environment.EER_experiment import Environment
    env = Environment()
    state_all = [list(range(0, 200))]
    action_all = env.act_space
    rl_brain = Sarsa(action_all, state_all)
    for i in range(500):
        done = False
        s = env.reset()
        act = rl_brain.choose_action(s)
        while not done:
            s_, r, done = env.step(act)
            act_ = rl_brain.choose_action(s)
            rl_brain.learn(s, act, r, s_, act_, done)
            act = act
            s = s_
    print(rl_brain._state_action_value)
    print(np.argmax(rl_brain._state_action_value, axis=0) + 1)