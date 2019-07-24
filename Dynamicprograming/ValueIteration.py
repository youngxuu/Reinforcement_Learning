#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn

from Dynamicprograming.Iterativepolicyevaluation import BaseDynamicPrograming
import numpy as np
import itertools
from scipy.sparse import csr_matrix, coo_matrix
import abc


class ValueIteration(BaseDynamicPrograming, metaclass=abc.ABCMeta):
    """
    value iteration algorithm for dynamic programing
    see pseudo code: Reinforcement learning an introduction pp 67
    :param
        state_all: list, a list of all possible state
        action_all: list a list of all possible action,default one dimension of action
        gamma: float, 0 < gamma< 1, reward discount rate
        tol: float, for every epoch, we calculate the sum of absolute value of delta state value (delta)
             and check whither state value is stable. if stable (i.e. delta < tol) then we stop the iteration
        sweeps: int, number of sweeps through the state set
    """
    def __init__(self, state_all, action_all, gamma, tol, sweeps):
        self.state_all = state_all
        self.action_all = action_all
        self.n_action = len(action_all)
        self.gamma = gamma
        self.tol = tol
        self.n_state = len(state_all)
        self.v = np.zeros(shape=[self.n_state, 1], dtype=np.float32)
        self.sweeps = sweeps
        self.policy = {}

    def _learn(self, **kwargs):
        for epoch in range(self.sweeps):
            opt_policy = None
            for idx, state in enumerate(self.state_all):
                delta = 1
                # print('value iteration for state: ', state)
                while self.tol < delta:
                    v = self.v[idx]
                    prob, r = self.state_a_prob(state)
                    prob = np.squeeze(prob)
                    prob_v = prob * (r + self.v)
                    tmp_value = np.sum(prob_v, axis=0)
                    self.v[idx] = np.max(tmp_value)
                    opt_policy = np.argmax(tmp_value)
                    delta = abs(v - self.v[idx])
                if epoch == self.sweeps - 1:
                    self.policy.update({idx: opt_policy})

    @abc.abstractmethod
    def state_a_prob(self, state):
        """
        :param state:
        :return: p(s_, r| s, a) w.r.t current state. numpy.array type, 3 dimensions [s, s_, a]
                 with the first dim == 1 (the current state !)
                 r w.r.t all states, numpy.array, 2 dimensions [num_state, 1]
                 here we only consider the simplest cases, ie reward only depends on
                  state: r = r(s_)
        """


class GamblerValueIteration(ValueIteration):
    """
     Gambler’s Problem :
     see ref: Reinforcement learning an introduction pp 68

     A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips.
     If the coin comes up heads, he wins as many dollars as he has staked on that flip;
     if it is tails, he loses his stake. The game ends when the gambler wins by reaching his goal of $100,
     or loses by running out of money. On each ﬂip, the gambler must decide what portion of his capital to stake,
     in integer numbers of dollars. This problem can be formulated as an undiscounted, episodic, finite MDP.
     The state is the gambler’s capital, s ∈{1,2,...,99} and the actions are stakes, a ∈{0,1,...,min(s,100−s)}.
     The reward is zero on all transitions except those on which the gambler reaches his goal, w
     hen it is +1. The state-value function then gives the probability of winning from each state.
     A policy is a mapping from levels of capital to stakes.
     The optimal policy maximizes the probability of reaching the goal.
     Let ph denote the probability of the coin coming up heads. If ph is known,
     then the entire problem is known and it can be solved, for instance, by value iteration.
    """
    def __init__(self, p_head=0.4, **kwargs):
        super().__init__(**kwargs)
        self.p_head = p_head
        self.p_tail = 1 - p_head
        self.state_idx = {s: idx for idx, s in enumerate(self.state_all)}

    def state_a_prob(self, state):
        possible_actions = [act for act in range(1, min(state, 100 - state) + 1)]
        if len(possible_actions) == 0:
            second_idx = [self.state_idx[state]]
            third_idx = [0]
            value = [0]
            act_s_prob = csr_matrix((value, (second_idx, third_idx)),
                                    shape=(self.n_state, self.n_action), dtype='float32').toarray()
            act_s_prob = act_s_prob[np.newaxis, :]
        else:
            pos_act_s_h = [state + act for act in possible_actions]
            pos_act_s_t = [state - act for act in possible_actions]
            value = [[self.p_head] * len(possible_actions)] + [[self.p_tail] * len(possible_actions)]
            third_idx = [idx for idx, _ in enumerate(possible_actions)] * 2
            second_idx = [self.state_idx[s] for s in pos_act_s_h]\
                         + [self.state_idx[s] for s in pos_act_s_t]
            value = np.array(list(itertools.chain(*value)))
            second_idx = np.array(second_idx)
            third_idx = np.array(third_idx)
            act_s_prob = csr_matrix((value, (second_idx, third_idx)),
                                    shape=(self.n_state, self.n_action), dtype='float32').toarray()
            act_s_prob = act_s_prob[np.newaxis, :]
        r = np.zeros(shape=(self.n_state, 1), dtype='float32')
        r[-1, 0] = 1
        return act_s_prob, r


if __name__ == '__main__':
    state_all = [i for i in range(0, 101)]
    action_all = [[j] for j in range(1, 51)]
    gamma = 0.9
    tol = 0.01
    values_list = []
    policy_list = []
    sweepss = [1, 2, 3, 32]
    for sweeps in sweepss:
        gamble = GamblerValueIteration(state_all=state_all, action_all=action_all, gamma=gamma, tol=tol, sweeps=sweeps)
        gamble.learn()
        values_list.append(np.squeeze(gamble.v).tolist())
        policy_list.append([act for act in gamble.policy.values()])
        print(policy_list)
    import matplotlib.pyplot as plt
    for idx, value in zip(sweepss, values_list):
        plt.plot(value[:-1], label='sweeps_' + str(idx))
    plt.ylim([-.2, 1])
    plt.xlim([0, 100])
    plt.show()
    plt.plot(policy_list[-1], label='policv')
    plt.ylim([0, 50])
    plt.xlim([0, 100])
    plt.show()


