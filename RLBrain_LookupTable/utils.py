#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/29 11:24
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : utils.py
# @Software: PyCharm


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
    return tuple([state_act_dir[i][state_i] for i, state_i in enumerate(state_act)])
