#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/1 20:49
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : BaseFunctionApproximation.py
# @Software: PyCharm


import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.linear_model import __all__
import abc


def feature_transform(org_feats, trans_tpye):
    std_feats = StandardScaler().fit_transform(org_feats)
    if trans_tpye == 'linear':
        feats = np.hstack((np.array([1]), std_feats))
    elif trans_tpye == 'rbf':
        feats = np.exp(np.sqrt(std_feats))
    else:
        feats = PolynomialFeatures(degree=2, include_bias=False).fit_transform(std_feats)
    return feats


class _BaseFuncApproximation(object, metaclass=abc.ABCMeta):
    """
    base rl brain for linear function approximation!
    :parameters
    ----------
    action_all: list, all the possible actions an agent can take.
    gamma: float, 0<gamma<1, reward discount rate
    epsilon: float, 0<epsilon<1, exploration rate
    feature_trans: string, transform the original features into higher feature space.
                    only 'linear', 'rbf', 'polynomials' aavailable. if none, default 'linear'

    """
    def __init__(self, action_all, gamma=0.1, epsilon=0.1, feature_trans=None):
        super(_BaseFuncApproximation, self).__init__()
        if not isinstance(action_all, list):
            raise TypeError("action_all only list type valid!!!")
        self.act_dir = {action: i for i, action in enumerate(action_all)}
        self.act_dim = len(action_all)
        if not isinstance(gamma, float):
            raise TypeError("Threshold is not a float value!")
        if not 0 <= gamma <= 1:
            raise ValueError("Threshold must >= 0 and <= 1!")
        self.gamma = gamma
        if not isinstance(epsilon, float):
            raise TypeError("Threshold is not a float value!")
        if not 0 <= epsilon <= 1:
            raise ValueError("Threshold must >= 0 and <= 1!")
        self.epsilon = epsilon
        if feature_trans:
            if not feature_trans.lower() in ['linear', 'rbf', 'polynomials']:
                raise ValueError("feature_trans must in ['linear', 'rbf', 'polynomials']")
            else:
                self.feat_trans = feature_trans
        else:
            self.feat_trans = 'linear'

    @abc.abstractmethod
    def choose_action(self, s):
        """choose an optimal action based on current state and policy"""

    @abc.abstractmethod
    def learn(self, **kwargs):
        """update action value function based on specific rl algorithms"""
