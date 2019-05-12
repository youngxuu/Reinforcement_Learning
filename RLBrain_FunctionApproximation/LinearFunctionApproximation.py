#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/8 18:36
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : LinearFunctionApproximation.py
# @Software: PyCharm

from RLBrain_FunctionApproximation.BaseFunctionApproximation import _BaseFuncApproximation, feature_transform
import numpy as np
from sklearn.linear_model import SGDRegressor
from RLBrain_FunctionApproximation.SGD import GradientDescent


class GradientMonteCarlo(_BaseFuncApproximation):

    def __init__(self, action_all, gamma=.9, epsilon=0.1, feature_trans=None):
        super().__init__(action_all, gamma=gamma, epsilon=epsilon,
                         feature_trans=feature_trans)
        self.ep_obs = []
        self.ep_as = []
        self.ep_rs = []

    def store_transactions(self, state, action, reward):
        """
        store state, action and reward in an episode.
        :param state: list of state
        :param action: action
        :param reward: reward, float.
        :return: None
        """
        self.ep_obs.append(state)
        self.ep_as.append(self.act_dir[action])
        self.ep_rs.append(reward)

    def _function_unfit(self):
        return SGDRegressor(max_iter=100, tol=0.001, penalty="l2", fit_intercept=False,
                            verbose=0,  warm_start=True)

    def learn(self, **kwargs):
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype='float32')
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        s_a = [self.ep_obs[t] + [(self.ep_as[t])] for t in range(len(self.ep_as))]
        s_a = np.array(s_a, dtype='float32')
        loss = self._function_update(s_a, discounted_ep_rs)
        self.ep_obs = []
        self.ep_as = []
        self.ep_rs = []
        return loss


if __name__ == '__main__':
    import time
    from Environment.EER_experiment import Environment
    env = Environment()
    state_all = [list(range(0, 200))]
    action_all = env.act_space
    rl_brain = GradientMonteCarlo(action_all, feature_trans='linear')
    t1 = time.time()
    loss_total = []
    for i in range(1000):
        done = False
        s = env.restart()
        while not done:
            act = rl_brain.choose_action(s)
            s_, r, done = env.step(act)
            rl_brain.store_transactions(s, act, r)
            s = s_
        loss = rl_brain.learn()
        if i % 100 == 0:
            print('EPOCH %d, total cost %0.2f, loss %.2f' % (i, time.time()-t1, loss))
            t1 = time.time()
            loss_total.append(loss)
    import matplotlib.pyplot as plt
    plt.plot(loss_total)
    plt.show()
    for i in range(200):
        print(rl_brain.function_predict([i]))






