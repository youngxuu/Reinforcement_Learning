#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
import os
import numpy as np
import tensorflow as tf
from RLBrain_DeepRL.BaseDeepRL import _BaseDeepRL

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
log_dir = 'D:/python_demo/Reinforcement_Learning/logs/'


class DeepQLearning(_BaseDeepRL):
    """
    an toy example of deep Q learning; with vector state and 1 hidden layer network
    params:
    n_features: int, the state dim, number of dimensions of state
    n_actions: int, number of unique actions
    actions: list, list of all possible actions
    learning_rate: float, 0<learning_rate<1, learning rate of evaluate network
    e_greedy: float, 0<e_greedy<1, exploration rate of agent
    reward_decay: float, 0<reward_decay<1, reward decay rate
    replace_target=100: int, the frequency of updating the target network
    replay_size=1000: int, size of replay memory
    batch_size=36: int, minibatch size, number of samples sampling from replay memory
    output_graph=False: bool, whether output the tensorflow graph.
    """
    def __init__(self, n_features, n_actions, actions,
                 learning_rate, e_greedy, reward_decay,
                 replace_target=100, replay_size=1000,
                 batch_size=36, output_graph=False):
        super().__init__(n_features, n_actions,
                         learning_rate, e_greedy, reward_decay)
        # actions index dict(index actions dict)
        self.act_idx = {act: idx for idx, act in enumerate(actions)}
        self.idx_act = {idx: act for act, idx in self.act_idx.items()}
        self.replace_target_iter = replace_target
        self.memory_size = replay_size
        # number of transactions saved since started
        self.transactions_count = 0
        self.replay_memory = np.zeros(shape=(self.memory_size, 2*self.n_features+3),
                                      dtype='float32')
        self.batch_size = batch_size
        self.output_graph = output_graph
        self._build_networks()

        # params for target network and eval network
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evaluate_network')
        with tf.variable_scope('hard_replacement'):
            # assign the learned params to target network
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        # gpu configs
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=config)
        # output graph to logdir
        if self.output_graph:
            if os.path.exists(log_dir):
                for file in os.listdir(log_dir):
                    os.remove(os.path.join(log_dir, file))
            tf.summary.FileWriter(log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        # number of learning iter since started(used for update target networks)
        self.learn_iter = 0

    def _build_networks(self,  **kwargs):
        # placeholders
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features, ], name='state')
        self.state_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features, ], name='state_')
        self.action = tf.placeholder(dtype=tf.int32, shape=[None, ], name='action')
        self.q_target_value = tf.placeholder(dtype=tf.float32, shape=[None, ], name='q_target_value')
        h1_dim = 100

        # target network
        with tf.variable_scope('target_network'):
            h1 = tf.layers.dense(self.state_, units=h1_dim, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.constant_initializer(0),
                                 name='hidden_1'
                                 )
            self.q_target = tf.layers.dense(h1, units=self.n_actions, activation=tf.nn.relu,
                                            kernel_initializer=tf.random_normal_initializer(),
                                            bias_initializer=tf.constant_initializer(0),
                                            name='q_target'
                                            )
        # eval network
        with tf.variable_scope('evaluate_network'):
            s1 = tf.layers.dense(self.state, units=h1_dim, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.constant_initializer(0.),
                                 name='hidden_1'
                                 )
            self.q_eval = tf.layers.dense(s1, units=self.n_actions, activation=tf.nn.relu,
                                          kernel_initializer=tf.random_normal_initializer(),
                                          bias_initializer=tf.constant_initializer(0.),
                                          name='q_eval')

        with tf.variable_scope('loss'):
            # idx for indexing the corresponding action q value
            a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action],
                                 axis=1)
            self.q_a_eval = tf.gather_nd(params=self.q_eval, indices=a_indices, name='q_a_eval')
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_a_eval, self.q_target_value, name='TD_error'))

        with tf.variable_scope('train'):
            self.train = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def choose_action(self, state):
        """
        choose actions based e_greedy,
        :param state:
        :return:
        """
        state = state[np.newaxis, :].astype('float32')
        if np.random.uniform(0, 1) < 1 - self.e_greedy:
            q_a = self.sess.run(self.q_target, feed_dict={self.state_: state})
            action_idx = np.argmax(q_a)
        else:
            action_idx = np.random.randint(0, self.n_actions)
        action = self.idx_act[action_idx]
        return action

    def learn(self, **kwargs):
        """
        q learning process;
        if transactions_count > replay memory size, then update the evaluate network based on
        TD error
        if learning count % replace target iter, then update the target network
        :param kwargs:
        :return:
        """
        if self.transactions_count >= self.memory_size:
            # update the target network
            if self.learn_iter % self.replace_target_iter:
                print('update target network')
                self.sess.run(self.target_replace_op)
            # sample a mini batch from replay memory
            sample = np.random.randint(0, self.memory_size, size=self.batch_size)
            sample_transactions = self.replay_memory[sample, :]
            sample_s = sample_transactions[:, :self.n_features]
            sample_s_ = sample_transactions[:, self.n_features:2*self.n_features]
            sample_a = sample_transactions[:, 2*self.n_features]
            sample_r = sample_transactions[:, 2*self.n_features+1]
            sample_done = sample_transactions[:, 2*self.n_features+2]
            # calculate the target q value
            y = sample_r
            for i, d in enumerate(sample_done):
                if d == 0:
                    y[i] = sample_r[i] + self.reward_decay * \
                           np.max(self.sess.run(self.q_target,
                                                feed_dict={self.state_: sample_s_[i, :][np.newaxis, :]}))

            # training
            self.sess.run([self.train, self.loss], feed_dict={
                self.state: sample_s, self.q_target_value: y, self.action: sample_a
            })
            self.learn_iter += 1

    def store_transactions(self, state, action, state_, reward, done):
        """
        storing the transactions and
        transform action to action index, done to 1(0)
        if replay memory is full
        overwrite the old memory by new one
        :param state:
        :param action:
        :param state_:
        :param reward:
        :param done:
        :return:
        """
        done_ = 1 if done else 0
        index = self.transactions_count % self.memory_size
        action = self.act_idx[action]
        self.replay_memory[index, :] = np.hstack((state, state_, [action, reward, done_]))
        self.transactions_count += 1


if __name__ == '__main__':
    rl_brain = DeepQLearning(n_features=10, n_actions=2, actions=['1', '2'],
                             learning_rate=0.001, e_greedy=0.1,
                             reward_decay=0.9, output_graph=True)
    # help(rl_brain)
    for i in range(1500):
        state = np.random.randint(1, 100, size=(10, ))
        a = rl_brain.choose_action(state)
        r = np.random.randint(1, 5, 1)
        state_ = np.random.randint(1, 100, size=(10, ))
        done = np.random.randint(0, 2, 1)
        rl_brain.store_transactions(state, a, state_, r, done)
        if done == 1:
            state = np.random.randint(1, 100, size=(10, ))
        else:
            state = state_
        rl_brain.learn()





