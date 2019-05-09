#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn

import abc


class _BaseDeepRL(object, metaclass=abc.ABCMeta):
    """
    base class for deep rl
    params:
    n_features: int, the state dim, number of dimensions of state
    n_actions: int, number of unique actions
    learning_rate: float, 0<learning_rate<1, learning rate of evaluate network
    e_greedy: float, 0<e_greedy<1, exploration rate of agent
    reward_decay: float, 0<reward_decay<1, reward decay rate
    """
    def __init__(self, n_features, n_actions,
                 learning_rate, e_greedy, reward_decay):
        super().__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.e_greedy = e_greedy
        self.reward_decay = reward_decay

    @abc.abstractmethod
    def choose_action(self, **kwargs):
        """
        choose action baesd on the current state.
        """

    @abc.abstractmethod
    def _build_networks(self,  **kwargs):
        """
        build the deep networks architectures needed.
        """

    @abc.abstractmethod
    def learn(self, **kwargs):
        """
        update the deep networks based on different DRL algorithms
        """
