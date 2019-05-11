#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn

import abc


class _BaseEnvironment(object, metaclass=abc.ABCMeta):
    """
    base environment
    """
    @abc.abstractmethod
    def reset(self):
        """
        reset the environment to initial state
        :return:
        """

    @abc.abstractmethod
    def step(self, **kwargs):
        """
        agent take an action, the environment gives a feed back (next state, reward and done)
        :return:
        """