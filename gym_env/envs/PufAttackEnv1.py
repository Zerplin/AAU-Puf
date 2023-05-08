import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding


class PufAttackEnv1(gym.Env):
    def __init__(self, render_mode=None, data='challenge_response_10k.csv'):
        self._df = pd.read_csv(data)
        for col in self._df.columns.values:
            self._df[col] = self._df[col].astype('int64')

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
        self._tries = 0

        # Choose the agent's location uniformly at random
        self._challenge = self._df.sample()
        return np.cumprod(np.fliplr(self._challenge.drop('64', axis=1)), axis=1, dtype=np.int8)[0]

    def step(self, action):
        self._tries += 1
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._challenge['64'], [((2 * action) - 1)])
        reward = (1 / self._tries) if terminated else 0  # Binary sparse rewards
        # return observation, reward, terminated, False, info
        return np.cumprod(np.fliplr(self._challenge.drop('64', axis=1)), axis=1, dtype=np.int8)[
            0], reward, terminated, {}
