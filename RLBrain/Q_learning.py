#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/31 19:19
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : Q_learning.py
# @Software: PyCharm


from RLBrain.utils import state_index, state_action_index
from RLBrain.Base_Brain import BaseBrain
import numpy as np


class QLearning(BaseBrain):
    """
   an implement of Q learning algorithm
    """

    def __init__(self, action_all, state_all, gamma=0.01, epsilon=0.1, alpha=0.1):
        super().__init__(action_all, state_all, gamma, epsilon)
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

    def learn(self, state, action, reward, state_, done):
        # print(state)
        state_act_index = state_action_index(action, state, self.act_dict, self.state_all)
        max_state_value = np.max(self._state_action_value, axis=0)
        if done:
            q_target = reward
        else:
            state_idx_ = state_index(state_, self.state_all)
            q_target = reward + self.alpha * max_state_value[state_idx_]
        self._state_action_value[state_act_index] += \
            self.alpha * (q_target - self._state_action_value[state_act_index])


if __name__ == '__main__':
    from Environment.EER_experiment import Environment
    env = Environment()
    state_all = [list(range(0, 200))]
    action_all = env.act_space
    rl_brain = QLearning(action_all, state_all)
    for i in range(500):
        done = False
        s = env.restart()
        while not done:
            act = rl_brain.choose_action(s)
            s_, r, done = env.step(act)
            rl_brain.learn(s, act, r, s_, done)
            s = s_
    print(rl_brain._state_action_value)
    print(np.argmax(rl_brain._state_action_value, axis=0) + 1)
