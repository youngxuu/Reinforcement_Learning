#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 18:27
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : SGD.py
# @Software: PyCharm


import numpy as np
from RLBrain_FunctionApproximation.BaseFunctionApproximation import feature_transform
from sklearn.linear_model import SGDRegressor


class GradientDescent(object):

    def __init__(self, trans_type, max_iter, lr):
        self.coef_ = None
        self.trans_type = trans_type
        self.max_iter = max_iter
        self.lr = lr

    def __str__(self):
        return 'GradientDescent(trans_type=%s, max_iter=%d, lr=%f)' % \
               (self.trans_type, self.max_iter, self.lr)

    __repr__ = __str__

    def _fit(self, x, y, verbs=False):
        x_trans = feature_transform(x, trans_tpye=self.trans_type)
        coef = self.coef_
        if coef is None:
            coef = np.random.random((x_trans.shape[-1], 1))
        cost = np.linalg.norm(y - np.dot(x_trans, coef), 2)

        for i in range(self.max_iter):
            for j in range(len(x_trans)):
                x_j = x_trans[j].reshape(1, -1)
                loss = (y[j] - np.dot(x_j, coef))[0]
                coef += self.lr * loss * x_j.T
            cost_i = np.linalg.norm(y - np.dot(x_trans, coef), 2)

            if verbs:
                if i % 10 == 0:
                    cost_i = np.linalg.norm(y - np.dot(x_trans, coef), 2)
                    print('iteration %d, mean square loss %f' % (i, cost_i))

            if np.abs(cost_i - cost) < 0.001:
                break
            cost = cost_i
        self.coef_ = coef

        return self

    def fit(self, x, y, verbs=False):
        return self._fit(x, y, verbs=verbs)

    def _predict(self, x):
        x_trans = feature_transform(x, trans_tpye=self.trans_type)
        if self.coef_ is None:
            raise Exception('model has not been fitted!!!')
        else:
            return np.dot(x_trans, self.coef_)

    def predict(self, x):
        return self._predict(x)
