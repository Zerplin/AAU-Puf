import math

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from pypuf.simulation import ArbiterPUF

# Settings
challenge_bit_length = 8
seed = 1337
# Use recommended granularity as presented by Ganji et al. 2016
M_delay_granularity = 10*math.sqrt(challenge_bit_length)/2.5
# Round to nearest even number for decision process to be correct
M_delay_granularity = round(M_delay_granularity / 2) * 2
print("M_granularity:",M_delay_granularity)
#M_delay_granularity = 32


class ArbiterPufDelayII(gym.Env):
    def __init__(self, render_mode=None):
        self._challenge = None
        self.puf = ArbiterPUF(n=challenge_bit_length, seed=seed)
        self.puf_stage = 0
        self.accumulated_delay_delta = 0

        self.observation_space = spaces.MultiBinary(challenge_bit_length + 3)

        self.action_space = spaces.Discrete(M_delay_granularity)
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
        self._challenge = (2 * self.np_random.randint(0, 2, (1, challenge_bit_length), dtype=np.int8) - 1)
        # cumulative product of the challenge for effectiveness

        return np.concatenate((np.cumprod(np.fliplr(self._challenge), axis=1, dtype=np.int8)[0],
                               [self._challenge[0][self.puf_stage]], [self.puf_stage], [self.accumulated_delay_delta]))

    def step(self, action):
        # Update
        self.accumulated_delay_delta += action
        self.puf_stage += 1

        terminated = False
        reward = 0
        next_observation = np.concatenate((np.cumprod(np.fliplr(self._challenge), axis=1, dtype=np.int8)[0], [self._challenge[0][self.puf_stage]], [self.puf_stage],
                                           [self.accumulated_delay_delta]))
        # An episode/challenge-walk is done when the agent has reached the end of the puf stages:
        if self.puf_stage == (challenge_bit_length - 1):
            # calculate the actual action based on the accumulated delay (top half of M, action=1, otherwise a=0)
            evaluated_action = 1 if self.accumulated_delay_delta >= M_delay_granularity / 2 else 0
            action_success = np.array_equal(self.puf.eval(self._challenge), [((2 * evaluated_action) - 1)])
            reward = 1 if action_success else 0
            terminated = True

        return next_observation, reward, terminated, {}
