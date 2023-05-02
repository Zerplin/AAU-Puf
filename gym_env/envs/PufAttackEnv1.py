import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from pypuf.simulation import XORArbiterPUF


class PufAttackEnv1(gym.Env):
    def __init__(self, render_mode=None):
        self.puf = XORArbiterPUF(n=64, k=4, seed=1337)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Dict({"agent": spaces.MultiBinary(64), "target": spaces.MultiBinary(64), })
        self.observation_space = spaces.MultiBinary(64)

        self.action_space = spaces.Discrete(2)
        self.render_mode = None
        self.viewer = None
        self.window = None
        self.clock = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, options=None):
        # We need the following line to seed self.np_random

        # Choose the agent's location uniformly at random
        self._challenge = (2 * self.np_random.randint(0, 2, (1, 64), dtype=np.int8) - 1)
        return np.cumprod(np.fliplr(self._challenge), axis=1, dtype=np.int8)[0]

    def step(self, action):
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self.puf.eval(self._challenge), [((2 * action) - 1)])
        reward = 1 if terminated else 0  # Binary sparse rewards
        # return observation, reward, terminated, False, info
        return np.cumprod(np.fliplr(self._challenge), axis=1, dtype=np.int8)[0], reward, terminated, {}
