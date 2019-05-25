#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn

from Dynamicprograming.Iterativepolicyevaluation import BaseDynamicPrograming
import numpy as np
import itertools
from scipy.sparse import csr_matrix


class PolicyIteration(BaseDynamicPrograming):
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
        self.policy_prob = (1 / self.n_act) * np.ones((self.n_state, self.n_act))
        self.n_epochs = n_epochs
        self.record_delta = 10
        self.deltas = []
        self.verbose = True
        self.epsion = 0.01

    def _value_iteration(self, ):
        r_policy = np.sum(self.policy * self.s_a_r, axis=1).reshape(-1, 1)
        p_policy = np.sum(self.policy * self.state_a_prob, axis=2)
        v_update = r_policy + self.gamma * np.dot(p_policy, self.value)
        delta = np.sum(np.abs(v_update - self.value))
        self.value = v_update.copy()
        return delta

    def _policy_iteration(self):
        value = self.value.copy().T
        value = np.expand_dims(value, axis=2)
        values = np.concatenate(tuple([value for acm in range(self.n_act)]), axis=2)
        p_dot_v = self.s_a_r + np.sum(self.state_a_prob * values, axis=1)
        max_idx = [np.argwhere(p_dot_v[i] == np.max(p_dot_v[i]))[:, 0].tolist() for i in range(p_dot_v.shape[0])]
        rows = [[i] * len(idxs) for i, idxs in enumerate(max_idx)]
        policyvalue = [(np.ones(shape=(len(idxs))) * 1/len(idxs)).tolist() for i, idxs in enumerate(max_idx)]
        rows = list(itertools.chain(*rows))
        columns = list(itertools.chain(*max_idx))
        policyvalue = list(itertools.chain(*policyvalue))
        policy = csr_matrix((policyvalue, (rows, columns)), shape=[self.n_state, self.n_act],
                            dtype='float32').todense()
        policy = policy.A
        if (policy == self.policy).all():
            return True
        else:
            self.policy = policy.copy()
            return False

    def _learn(self):
        """
        value iteration and policy iteration
        :return:
        """
        stable_count = 0
        for epoch in range(self.n_epochs):
            delta = self._value_iteration()
            policy_stable = self._policy_iteration()
            if epoch % self.record_delta == 0:
                self.deltas.append(delta)
                if self.verbose:
                    print('epoch: %d' % epoch, 'delta: ', delta)
            if policy_stable and stable_count == 0:
                print('iteration policy_stable at epoch: ', epoch)
                stable_count += 1

            if policy_stable and delta <= self.tol:
                print('iteration stopped with convergence')
                break


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
    policyevaluation = PolicyIteration(state_a_prob=transition_matrix,
                                       gamma=0.9, tol=0.001,
                                       s_a_r=state_act_reward,
                                       n_epochs=500)
    policyevaluation.learn()
    print('state_value: ', policyevaluation.value)
    print('policy: ', policyevaluation.policy)

