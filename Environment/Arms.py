#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn


import pyglet
import numpy as np
from Environment.BaseEnvironment import _BaseEnvironment


class ArmEnv(_BaseEnvironment):
    viewer = None
    dt = 0.1
    action_bound = [-1, 1]
    goal = {'x': np.random.randint(0, 400), 'y': np.random.randint(0, 400), 'l': 40}
    state_dim = 9
    action_dim = 2

    def __init__(self):
        self.goal = {'x': np.random.randint(0, 380), 'y': np.random.randint(0, 380), 'l': 40}
        self.arm_info = np.zeros(
            2, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 110
        self.arm_info['r'] = np.pi / 6
        self.on_goal = 0

    def step(self, action):
        done = False
        r = 0.
        # 计算单位时间 dt 内旋转的角度, 将角度限制在360度以内
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2  # normalize
        # 如果手指接触到蓝色的 goal, 我们判定结束回合 (done)
        # 所以需要计算 finger 的坐标
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
        r = -np.sqrt(dist2[0] ** 2 + dist2[1] ** 2)
        if self.goal['x'] - self.goal['l'] / 2 < finger[0] < self.goal['x'] + self.goal['l'] / 2:
            if self.goal['y'] - self.goal['l'] / 2 < finger[1] < self.goal['y'] + self.goal['l'] / 2:
                r += 1. * self.on_goal  # finger 在 goal 以内
                self.on_goal += 1
                if self.on_goal > 50:
                    done = True
        else:
            self.on_goal = 0
        s = np.concatenate((a1xy_ / 200, finger / 200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s, r, done

    def reset(self):

        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
        self.on_goal = 0
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
        # state
        s = np.concatenate((a1xy_ / 200, finger / 200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(2) - 0.5  # two radians


class Viewer(pyglet.window.Window):
    bar_thc = 5  ##手臂厚度

    def __init__(self, arm_info, goal):
        # 添加 arm 信息
        super(Viewer, self).__init__(width=400,
                                     height=400, resizable=False,
                                     caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)  # 背景颜色
        self.arm_info = arm_info
        # 添加窗口中心点, 手臂的根
        self.center_coord = np.array([200, 200])
        # 蓝色 goal 的信息包括他的 x, y 坐标, goal 的长度 l
        self.batch = pyglet.graphics.Batch()
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))  # color

        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,
                     250, 300,
                     260, 300,
                     260, 250]), ('c3B', (249, 86, 86) * 4,))
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,  # location
                     100, 160,
                     200, 160,
                     210, 150]), ('c3B', (249, 86, 86) * 4,))

    def render(self):  # 刷新屏幕
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

        pass

    def on_draw(self):
        self.clear()  # 刷新手臂位置
        self.batch.draw()

    def _update_arm(self):
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = self.center_coord  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        # 第一段手臂的4个点信息
        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        # 第二段手臂的4个点信息
        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        # 将点信息都放入手臂显示中
        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
        pass  # 更新手臂位置信息


if __name__ == '__main__':
    import time
    env = ArmEnv()
    state_dim = ArmEnv.state_dim
    action_dim = ArmEnv.action_dim
    a_slice = [-1 + i*0.25 for i in range(0, 9)]
    action_all = []
    for a1 in a_slice:
        for a2 in a_slice:
            action_all.append([a1, a2])
    from RLBrain_DeepRL.DeepQLearning import DeepQLearning
    rl_brain = DeepQLearning(n_features=state_dim, n_actions=len(a_slice)**2,
                             actions=action_all,
                             learning_rate=0.001, e_greedy=0.2,
                             reward_decay=0.9, output_graph=True, double=True, EnQ=True)
    t0 = time.time()
    ep_r = np.zeros(10)
    for epoch in range(400):
        finished = False
        state = env.reset()
        r_total = 0.
        for t in range(100):
            env.render()
            act = rl_brain.choose_action(state)
            # print(act)
            state_, r, finished = env.step(act)
            rl_brain.store_transactions(state, act, state_, r, finished)
            r_total += r
            rl_brain.learn()
            state = state_
            if finished:
                print('on target')
                break
        index = epoch % 10
        ep_r[index] = r_total
        sum_r = np.sum(ep_r)
        # if epoch % 10 == 0:
        t1 = time.time()
        print('epoch: %d;    learning iter: %d;    time cost: %.2f'
              % (epoch, rl_brain.learn_iter, t1 - t0))
        print('total reward: %.2f, total reward(last 10 epoch): %.2f' % (r_total, sum_r))
        t0 = t1

