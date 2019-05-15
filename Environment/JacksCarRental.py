#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn

import numpy as np
from Environment.BaseEnvironment import _BaseEnvironment


class CarRental(_BaseEnvironment):
    """
    an implement of jack's car rental
    States: Two locations, maximum of 20 cars at each
    Actions: Move up to 5 cars between locations overnight
    Reward: $10 for each car rented (must be available)
    Transitions: Cars returned and requested randomly
                Poisson distribution, n returns/requests
                1st location: average requests = 3, average returns = 3
                2nd location: average requests = 4, average returns = 2
                in case of not return all the the rented cars,
                we replace the distribution of returns,
                with uniformly return number of cars to loc1, the rest to loc2
    """

    def __init__(self,
                 n_locations=2,
                 rent_fee=10,
                 avg_requests=np.array([[3], [4]], dtype='float32'),
                 avg_return=np.array([[3], [2]], dtype='float32'),
                 avg_cars=10):
        self.n_locations = n_locations
        self.avg_requests = avg_requests
        self.avg_return = avg_return
        self.rent_fee = rent_fee
        self.total_cars = avg_cars * self.n_locations
        self.state = avg_cars * np.ones(shape=[self.n_locations, 1], dtype='float32')
        self.t = 0

    def reset(self):
        """
        reset the environment to initial state
        return the initial state
        :return:
        """
        self.state = (self.total_cars / self.n_locations) * \
                     np.ones(shape=[self.n_locations, 1], dtype='float32')
        self.t = 0

        return self.state.copy()

    def step(self, actions):
        """
        here we implement an toy step method with only one dim of action space.
        action is a scalar with (without negative sign) ,
        indicating the number of car need to be moved(and witch direction to move)
        if actions > 0, then we move the car in location 1 to location 2,
        otherwise, we move car in location 12 to location 2
        questions(important!!!)
        actions with one dims, what about actions with more than one dims??
        i.e. if number of locations is larger than 2,
        then how to define actions and validation the
        feasibility of actions.
        :param actions:
        :return:
        """
        self.t += 1
        done = False if self.t <= 30 else True
        #     check feasibility of actions
        if actions > 0:
            if self.state[0] < actions:
                actions = self.state[0, 0].copy()
        else:
            if self.state[1] + actions < 0:
                actions = -self.state[1, 0].copy()
        # update state after Transitions
        actions_all = np.array([[-actions], [actions]], dtype='float32')
        self.state += actions_all
        #    generate requests and returns
        requests = self._requests()
        self.state -= requests
        returns = self._returns(requests)
        self.state += returns
        reward = self.rent_fee * np.sum(requests)
        return self.state.copy(), reward, done

    def _requests(self):
        """
        generate requests and check feasibility of requests
        :return:
        """
        # generate requests
        requests = np.random.poisson(self.avg_requests, size=(self.n_locations, 1))
        # check requests
        for loc in range(self.n_locations):
            if requests[loc, 0] > self.state[loc, 0]:
                requests[loc, 0] = self.state[loc, 0].copy()
        return requests

    def _returns(self, requests):
        """
        generate returns
        note:  since the total car in system is
               equal to maximum of location capacity,
               no chick is needed for returns
        however, we need to check whither the total returns is equal the total rentals

        the same questions raised when the number of locations is large than 2
        :param requests:
        :return: returns
        """
        # total_returns = np.sum(requests) + 1
        # returns = [0, 0]
        # while total_returns != np.sum(requests):
        #     returns = np.random.poisson(self.avg_return, size=(self.n_locations, 1))

        total_rented = np.sum(requests).copy()
        # print(total_rented)
        if total_rented == 0:
            returns = np.zeros_like(self.state)
        else:
            returns0 = np.random.randint(0, total_rented, 1)[0]
            returns = np.array([[returns0], [total_rented-returns0]])

        return returns


if __name__ == '__main__':
    env = CarRental()
    for epoch in range(10):
        state = env.reset()
        done = False
        print('epoch %d' % epoch)
        while not done:
            act = np.random.randint(-env.total_cars, env.total_cars, 1)[0]
            state_, reward, done = env.step(act)
