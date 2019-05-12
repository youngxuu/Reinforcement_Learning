#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/1 20:49
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : BaseFunctionApproximation.py
# @Software: PyCharm


import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import abc


def feature_transform(org_feats, trans_tpye):

    if trans_tpye == 'linear':
        feats = np.hstack((np.ones(shape=(len(org_feats), 1)), org_feats))
    elif trans_tpye == 'rbf':
        feats = np.exp(np.sqrt(org_feats))
    elif trans_tpye == 'polynomials':
        org_feats = np.array(org_feats, dtype='float32')
        feats = PolynomialFeatures(degree=2, include_bias=True).fit_transform(org_feats)
    else:
        raise Exception('only linear, rbf and polynomials available')
    feats = feats/np.max(feats)
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
                    only 'linear', 'rbf', 'polynomials'  available. if none, default 'linear'

    """
    def __init__(self, action_all, gamma=0.1, epsilon=0.1, feature_trans=None):
        super(_BaseFuncApproximation, self).__init__()
        if not isinstance(action_all, list):
            raise TypeError("action_all only list type valid!!!")
        if isinstance(action_all[0], str):
            self.act_dir = {action: i for i, action in enumerate(action_all)}
        else:
            self.act_dir = {action: action for i, action in enumerate(action_all)}

        self.idx_act = {idx: action for idx, action in enumerate(action_all)}
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
        self.function = None

    @abc.abstractmethod
    def _function_unfit(self):
        """select a function type to approximate the action value function """

    def _function(self, s_a_trans):
        self.function = self._function_unfit()
        self.function.fit(s_a_trans, np.zeros((len(s_a_trans), ), dtype='float32'))

    def choose_action(self, s):
        """choose an optimal action based on current state and policy"""
        s_a = [s + [action] for action in (self.act_dir.values())]
        s_a = np.array(s_a, dtype='float32')
        s_a_trans = feature_transform(s_a, trans_tpye=self.feat_trans)
        if not self.function:
            act_idx = np.random.choice(range(self.act_dim), 1)[0]
            self._function(s_a_trans=s_a_trans)
        else:
            if np.random.uniform([0, 1], 1)[0] > 1 - self.epsilon:
                action_value = self.function.predict(s_a_trans)
                act_idx = np.argmax(action_value)
            else:
                act_idx = np.random.choice(range(self.act_dim), 1)[0]
        return self.idx_act[act_idx]

    def _function_update(self, x, y):
        x_trans = feature_transform(x, trans_tpye=self.feat_trans)
        self.function.fit(x_trans, y)
        y_pred = self.function.predict(x_trans)
        return mean_squared_error(y_pred, y)

    def function_predict(self, s):
        s_a = [s + [a] for a in list(self.idx_act.values())]
        x_trans = feature_transform(s_a, trans_tpye=self.feat_trans)
        return self.function.predict(x_trans)

    @abc.abstractmethod
    def learn(self, **kwargs):
        """update action value function based on specific rl algorithms"""
