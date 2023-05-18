import math

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from pypuf.simulation import ArbiterPUF


class ArbiterPufDelayIICRPs(gym.Env):
    def __init__(self, M_delay_granularity=0):
        self._df = pd.read_csv("FPGA1M_Aghaie+Moradi.csv")
        self.challenge_bit_length = 64
        # Use recommended granularity as presented by Ganji et al. 2016
        if M_delay_granularity == 0:
            self.M_delay_granularity = 10 * math.sqrt(self.challenge_bit_length) / 2.5
            # Round to nearest even number for decision process to be correct
            self.M_delay_granularity = round(self.M_delay_granularity / 2) * 2
        else:
            self.M_delay_granularity = M_delay_granularity
        print("M_granularity:", self.M_delay_granularity)
        self._challenge = None
        self.cumprod = None
        self.puf_stage = 0
        self.accumulated_delay_delta = 0

        # Default observations: [cumprod, challenge bit, i-stage, accumulated delay]
        self.observation_space = spaces.MultiBinary(self.challenge_bit_length + 3)

        self.action_space = spaces.Discrete(self.M_delay_granularity)
        self.render_mode = None
        self.viewer = None
        self.window = None
        self.clock = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, options=None):
        self.puf_stage = 0
        self.accumulated_delay_delta = 0
        # Choose a random challenge for observation
        self._challenge = self._df.sample()
        self.cumprod = np.cumprod(np.fliplr(self._challenge.drop('64', axis=1)), axis=1, dtype=np.int8)[0]
        # cumulative product of the challenge for effectiveness

        current_bit = self._challenge.drop('64', axis=1).to_numpy()[0][self.puf_stage]
        return np.concatenate((self.cumprod,
                               [current_bit], [self.puf_stage], [self.accumulated_delay_delta]))

    def step(self, action):
        # Update
        self.accumulated_delay_delta += action
        self.puf_stage += 1

        terminated = False
        reward = 0
        current_bit = self._challenge.drop('64', axis=1).to_numpy()[0][self.puf_stage]
        next_observation = np.concatenate((self.cumprod, [current_bit], [self.puf_stage],
                                           [self.accumulated_delay_delta]))
        # An episode/challenge-walk is done when the agent has reached the end of the puf stages:
        if self.puf_stage == (self.challenge_bit_length - 1):
            # calculate the actual action based on the accumulated delay (top half of M, action=1, otherwise a=0)
            evaluated_action = 1 if self.accumulated_delay_delta >= self.M_delay_granularity / 2 else 0
            action_success = np.array_equal(self._challenge['64'], [((2 * evaluated_action) - 1)])
            reward = 1 if action_success else 0
            terminated = True

        return next_observation, reward, terminated, {}
