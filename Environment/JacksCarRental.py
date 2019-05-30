#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn

import numpy as np
from Environment.BaseEnvironment import _BaseEnvironment


class CarRental(_BaseEnvironment):
    """
    an implement of jack's car rental
    Jack manages two locations for a nationwide car rental company.
    Each day, some number of customers arrive at each location to rent cars.
    If Jack has a car available, he rents it out and is credited $10
    by the national company.
    If he is out of cars at that location, then the business is lost.
    Cars become available for renting the day after they are returned.
    To help ensure that cars are available where they are needed,
    Jack can move them between the two locations overnight, at a cost of $2 per car moved.
    We assume that the number of cars requested and returned at each location are
    Poisson random variables, meaning that the probability that the number is n is λn/n! e−λ,
    where λ is the expected number.
    Suppose λ is 3 and 4 for rental requests at the first and second locations
    and 3 and 2 for returns.
    To simplify the problem slightly, we assume that there can be no more than 20 cars
    at each location (any additional cars are returned to the nationwide company,
    and thus disappear from the problem) and a maximum of five cars can be moved
    from one location to the other in one night. We take the discount rate to be
    γ = 0.9 and formulate this as a continuing finite MDP,
    where the time steps are days,
    the state is the number of cars at each location at the end of the day,
    and the actions are the net numbers of cars moved between the two locations overnight.
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
        self.loc_capacity = 20

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
        done = False if self.t <= 10 else True
        # check feasibility of actions
        if actions > 0:
            if self.state[0] < actions:
                actions = self.state[0, 0].copy()
            # print('number of cars moved to loc2: ', actions)
        else:
            if self.state[1] + actions < 0:
                actions = -self.state[1, 0].copy()
            # print('number of cars moved to loc1: ', actions)

        # update state after Transitions
        actions_all = np.array([[-actions], [actions]], dtype='float32')
        self.state += actions_all
        # generate requests and returns
        requests = self._requests()
        self._returns()
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
        requests = np.where(requests >= self.state,
                            self.state, requests)

        self.state -= requests

        return requests

    def _returns(self):
        """
        generate returns
        and check feasibility of returns
        :return: returns
        """

        returns = np.random.poisson(self.avg_return, size=(self.n_locations, 1))

        returns = np.where(returns + self.state >= self.loc_capacity,
                           self.loc_capacity - self.state, returns)

        self.state += returns


if __name__ == '__main__':
    env = CarRental()
    for epoch in range(1):
        state = env.reset()
        done = False
        print('epoch %d' % epoch)
        while not done:
            act = np.random.randint(-env.total_cars, env.total_cars, 1)[0]
            state_, reward, done = env.step(act)
