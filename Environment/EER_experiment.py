#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/28 20:41
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @File    : EER_experiment.py
# @Software: PyCharm


class Environment(object):
    """
    The environment is based on the experiment of a recently published paper.
    for more details of the experiment, please refer to:
    Hanaki N , Kirman A , Pezanis-Christou P . Observational and reinforcement pattern-learning: An exploratory study[J]
    . European Economic Review, 2018:S0014292118300187.

    """
    def __init__(self):
        self.count = 0
        self.act_space = [1, 2, 3, 4]

    def restart(self):
        self.count = 0
        ini_state = [self.count]
        return ini_state

    def step(self, act):
        self.count += 1
        done = False
        if not isinstance(act, int):
            raise TypeError('action must int type, but got %s instead.' % (type(act)))
        if act not in self.act_space:
            raise ValueError('action must in [1, 2, 3, 4], but got %s instead.' % (act))
        r = 0
        if act == 4:
            r = 0.4
        elif (3 - act) == (self.count - 1) % 3:
            r = 1
        state = [self.count]
        if self.count == 200:
            done = True
        return state, r, done


if __name__ == '__main__':
    from RLBrain_LookupTable.Monte_Carlo_Control import MonteCarloControl
    import numpy as np

    env = Environment()
    state_all = [list(range(0, 200))]
    action_all = env.act_space
    rl_brain = MonteCarloControl(action_all, state_all)
    for i in range(100):
        done = False
        s = env.restart()
        while not done:
            act = rl_brain.choose_action(s)
            s_, r, done = env.step(act)
            rl_brain.store_transactions(s, act, r)
            s = s_
        rl_brain.learn()
    print(rl_brain._state_action_value)
    print(np.argmax(rl_brain._state_action_value, axis=0) + 1)



