#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/6 12:30
# @Author  : xuyong
# @email   : xuyong@smail.swufe.edu.cn
# @Site    : 
# @File    : easy_21.py
# @Software: PyCharm


import numpy as np
import abc
from Environment.BaseEnvironment import _BaseEnvironment


class Easy21(_BaseEnvironment):
    """
    this is an 'easy-21' environment,
    it is a demo
    1. The game is played with an infinite deck of cards
    2. Each draw from the deck results in a value between 1 and 10
    (uniformly distributed) with a colour of red (probability 1/3) or
    black (probability 2/3).
    3. There are no aces or picture (face) cards in this game
    4. At the start of the game both the player and the dealer draw one black card (fully observed)
    5. Each turn the player may either stick or hit
        If the player hits then she draws another card from the deck
        If the player sticks she receives no further cards
    6. The values of the player’s cards are added (black cards) or subtracted (red cards)
    7. If the player’s sum exceeds 21, or becomes less than 1, then she “goes bust” and loses the game (reward -1)
    8. If the player sticks then the dealer starts taking turns.
         The dealer always sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes bust,
         then the player wins; otherwise, the outcome – win (reward +1), lose (reward -1), or draw (reward 0) –
         is the player with the largest sum.

    """
    def __init__(self):
        super().__init__()
        self.player_sum = 0
        self.dealer_sum = 0
        self.dealer_init_cards = None

    def reset(self):
        """
        Initialize the game,
        each player draw one black card(fully observed)
        return :initial state
        """
        dealer_init_cards, player_init_cards = np.random.choice(np.arange(1, 11), 2)
        self.dealer_init_cards = dealer_init_cards
        self.dealer_sum = dealer_init_cards
        self.player_sum = player_init_cards
        state = [player_init_cards, dealer_init_cards]
        return state

    def step(self, action):
        """
        player take an action(stick or hit)
        if stick ,dealer take actions according to the policy defined till the game ended
        if hit and player_sum > 21, game ended with R = -1
        if hit and player_sum <=0, game ended with R = -1
        if hit and  0 < player_sum < 21, return state, R=None
        :param action:
        :return: state, reward , done
        """
        act_dir = {1: 'hit', 0: 'stick'}
        if isinstance(action, str):
            if action.lower() in ['hit', 'stick']:
                action = action.lower()
            else:
                raise ValueError('invalid string action, only "hit" and "stick" allowed')
        if isinstance(action, int):
            if action == 1 or action == 0:
                action = act_dir[action]
            else:
                raise ValueError('invalid int action, only "0" and "1" allowed')

        reward = 0
        done = False
        if action == 'hit':

            card_color = np.random.choice(['red', 'black'], p=[2/3, 1/3])
            card_num = np.random.randint(1, 11)
            if card_color == 'red':
                self.player_sum += card_num
            else:
                self.player_sum -= card_num
            if self.player_sum > 21 or self.player_sum <= 0:

                done = True
                reward = -1
            state = [self.player_sum, self.dealer_init_cards]
            return state, reward, done
        else:

            while 0 < self.dealer_sum < 17:
                card_color = np.random.choice(['red', 'black'], p=[2 / 3, 1 / 3])
                card_num = np.random.randint(1, 11)
                if card_color == 'red':
                    self.dealer_sum += card_num
                else:
                    self.dealer_sum -= card_num

            state = [self.player_sum, self.dealer_init_cards]

            done = True
            if self.dealer_sum > 21 or self.dealer_sum <= 0:

                reward = 1
            else:
                if self.player_sum > self.dealer_sum:
                    reward = 1
                elif self.dealer_sum == self.player_sum:

                    reward = 0
                else:

                    reward = -1
            return state, reward, done


if __name__ == '__main__':
    env = Easy21()
    help(env)
    # for i in range(100):
    #     print('game start %d'%(i + 1))
    #     print('\n')
    #     done = False
    #     S = env.restart()
    #
    #     while not done:
    #         player_act = np.random.randint(2)
    #         print(player_act)
    #         S, R, done = env.step(player_act)
    #         print(S)












