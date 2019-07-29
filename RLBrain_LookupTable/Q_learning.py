#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/31 19:19
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : Q_learning.py
# @Software: PyCharm


from RLBrain_LookupTable.utils import state_index, state_action_index
from RLBrain_LookupTable.Base_Brain import BaseBrain
import numpy as np
import abc


class QLearning(BaseBrain):
    """
    an implement of Q learning algorithm
    """

    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        # print(self.state_all)
        # print(self.gamma)
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


class DoubleQLearning(QLearning, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state_action_value_v2 = np.zeros(self._state_action_dim, dtype='float32')

    def _get_next_idx(self, state_action_value_arr, state_, done):
        if done:
            return
        else:
            a_idx_v = np.unravel_index(np.argmax(state_action_value_arr),
                                       state_action_value_arr.shape)[0]
            action_v = self.idx_act[a_idx_v]
            state_act_index_v = state_action_index(action_v, state_,
                                                   self.act_dict, self.state_all)
            return state_act_index_v

    def learn(self, state, action, reward, state_, done):
        # state and action index
        state_act_index = state_action_index(action, state,
                                             self.act_dict, self.state_all)
        # argmax(a) w.r.t. state action value v1
        state_act_index_v1 = self._get_next_idx(self._state_action_value, state_, done)
        # argmax(a) w.r.t. state action value v2
        state_act_index_v2 = self._get_next_idx(self._state_action_value_v2, state_, done)
        if done:
            q_target = reward
            q_target_v2 = reward
        else:
            q_target = reward + self.alpha * self._state_action_value_v2[state_act_index_v1]
            q_target_v2 = reward + self.alpha * self._state_action_value[state_act_index_v2]
        if np.random.uniform([0, 1])[0] < 0.5:
            self._state_action_value[state_act_index] += \
                self.alpha * (q_target - self._state_action_value[state_act_index])
        else:
            self._state_action_value_v2[state_act_index] += \
                self.alpha * (q_target_v2 - self._state_action_value_v2[state_act_index])

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
            total_value = self._state_action_value_v2 + self._state_action_value
            act_idx = np.argmax(total_value, axis=0)[state_idx]
        else:
            act_idx = np.random.randint(self.act_num)
        return self.idx_act[act_idx]


if __name__ == '__main__':
    from Environment.EER_experiment import Environment
    env = Environment()
    state_all = [list(range(0, 200))]
    action_all = env.act_space
    rl_brain = DoubleQLearning(action_all=action_all, state_all=state_all)
    for i in range(500):
        done = False
        s = env.reset()
        while not done:
            act = rl_brain.choose_action(s)
            s_, r, done = env.step(act)
            rl_brain.learn(s, act, r, s_, done)
            s = s_
    print(rl_brain._state_action_value)
    print(np.argmax(rl_brain._state_action_value, axis=0) + 1)


