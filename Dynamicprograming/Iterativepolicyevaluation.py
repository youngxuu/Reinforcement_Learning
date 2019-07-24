#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn

import numpy as np


class BaseDynamicPrograming(object):
    """
    base class for dynamic programing
    """
    @staticmethod
    def _check_transition_p(state_a_p):
        """

        test whither the transition matrix is feasible
        :param state_a_p: numpy.array, 3 dimensions w.r.t [s, s_, a]
        :return:
        """
        p_sum = np.sum(state_a_p, axis=1)
        return (p_sum == 1).all()

    def _learn(self, **kwargs):
        """
        :param kwargs:
        :return:
        """

    def learn(self):
        self._learn()


class IterativePolicyEvaluation(BaseDynamicPrograming):
    """
    Iterative Policy Evaluation;
    see ref: Reinforcement learning an introduction pp 61
    params:
    state_a_prob: an numpy array with 3 dimensions (state, next state, action)
                with each action slice specifies the transition probability of
                current state to next state (i.e. transition matrix)
    gamma: float, 0 < gamma< 1, reward discount rate
    tol: float, for every epoch, we calculate the sum of absolute value of delta state value (delta)
         and check whither state value is stable. if stable (i.e. delta < tol) then we stop the iteration

    s_a_r: numpy array, with shape (n_state, n_action)
           with each element represent the reward received after take an action j on state i
    n_epochs: int, number of iterations
    reward_state_depend: bool default True, i.e. reward is only depend on state
    """
    def __init__(self, state_a_prob, gamma, tol, s_a_r, n_epochs,
                 ):
        self.gamma = gamma
        self.tol = tol
        self.n_state = state_a_prob.shape[0]
        self.n_act = state_a_prob.shape[2]
        if self._check_transition_p(state_a_prob):
            self.state_a_prob = state_a_prob
        else:
            raise ValueError('invalid  Transition matrix.')
        self.state_action_reward = s_a_r
        self.value = np.zeros(shape=(self.n_state, 1), dtype='float32')
        self.s_a_r = s_a_r
        self.policy = np.ones((self.n_state, self.n_act)) / self.n_act
        self.policy_prob = (1/self.n_act) * np.ones((self.n_state, self.n_act))
        self.n_epochs = n_epochs
        self.record_delta = 10
        self.deltas = []
        self.verbose = True

    def _learn(self):
        """
        update the state value by Bellman equation
        :return:
        """
        for epoch in range(self.n_epochs):
            r_policy = np.sum(self.policy * self.s_a_r, axis=1).reshape(-1, 1)
            p_policy = np.sum(self.policy * self.state_a_prob, axis=2)
            v_update = r_policy + self.gamma * np.dot(p_policy, self.value)
            delta = np.sum(np.abs(v_update - self.value))
            self.value = v_update.copy()
            if epoch % self.record_delta == 0:
                self.deltas.append(delta)
                if self.verbose:
                    print('epoch: %d' % epoch, 'delta: ', delta)
            if delta <= self.tol:
                print('iteration stopped with convergence')
                break
            elif epoch == self.n_epochs - 1:
                print('iteration stopped, but not convergence!')
            else:
                continue


if __name__ == '__main__':
    ''' 
    4×4 grid world:
    possible state: 1-14, and terminal state 15
    state reward: r=-1 with state 1-14, r=0 with state 15
    actions: up, down, right and left
    grid word
    15 1  2  3 
    4  5  6  7 
    8  9  10 11
    12 13 14 15
    '''
    from scipy.sparse import csr_matrix
    state_act_reward = -1 * np.ones(shape=(15, 4), dtype='float32')
    state_act_reward[0, 3] = 0.
    state_act_reward[3, 0] = 0.
    state_act_reward[10, 1] = 0.
    state_act_reward[13, 2] = 0.
    state_act_reward[14, :] = 0.

    row = np.arange(0, 15)
    print(row)
    col_up = np.array([0, 1, 2, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14])
    data = np.ones(15)
    p_up = csr_matrix((data, (row, col_up)), shape=[15, 15],
                      dtype='float32').todense()
    col_down = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 11, 12, 13, 14])
    p_down = csr_matrix((data, (row, col_down)), shape=[15, 15],
                        dtype='float32').todense()
    col_right = np.array([1, 2, 2, 4, 5, 6, 6, 8, 9, 10, 10, 12, 13, 14, 14])
    col_left = np.array([14, 0, 1, 3, 3, 4, 5, 7, 7, 8, 9, 11, 11, 12, 14])
    p_right = csr_matrix((data, (row, col_right)), shape=[15, 15],
                         dtype='float32').todense()
    p_left = csr_matrix((data, (row, col_left)), shape=[15, 15],
                        dtype='float32').todense()
    p = tuple([np.expand_dims(arr, axis=2) for arr in [p_up, p_down, p_right, p_left]])
    transition_matrix = np.concatenate(p, axis=2)
    policyevaluation = IterativePolicyEvaluation(state_a_prob=transition_matrix,
                                                 gamma=1, tol=0.001,
                                                 s_a_r=state_act_reward,
                                                 n_epochs=10000)
    policyevaluation.learn()
    print('state_value: ', policyevaluation.value)





