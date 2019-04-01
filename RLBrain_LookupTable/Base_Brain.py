#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/31 20:17
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : Base_Brain.py
# @Software: PyCharm


"""
base class for rl brain(lookup table)
"""
import numpy as np
import abc


class BaseBrain(object, metaclass=abc.ABCMeta):

    def __init__(self, action_all, state_all, gamma=0.01, epsilon=0.1):
        """
        base class for rl brain(lookup table)

        Parameters
        ----------
        action_all: list, all possible actions an agent have
        state_all: list, all possible state the RL environment have(look_up table)
        gamma: float, 0 < gamma < 1, reward discount rate
        epsilon: float, 0 < epsilon < 1, explore rate
        """
        super().__init__()
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
        if not isinstance(gamma, float):
            raise TypeError("Threshold is not a float value!")
        if not 0 < gamma < 1:
            raise ValueError("Threshold must >= 0 and <= 1!")
        self.gamma = gamma
        self._state_action_dim = [self.act_num] + self.state_dim
        self._state_action_value = np.zeros(self._state_action_dim, dtype='float32')
        self.epsilon = epsilon

    @abc.abstractmethod
    def choose_action(self, state):
        """choose an optimal action based on current state and policy"""

    @abc.abstractmethod
    def learn(self, **kwargs):
        """update action value function based on specific rl algorithms"""


