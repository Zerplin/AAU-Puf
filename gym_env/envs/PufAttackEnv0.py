import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from pypuf.simulation import ArbiterPUF


class PufAttackEnv0(gym.Env):
    def __init__(self, render_mode=None):
        # self.puf = XORArbiterPUF(n=64, k=4, seed=1337)
        self.puf = ArbiterPUF(n=64, seed=1337)

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
        self._challenge = (2 * self.np_random.randint(0, 2, (1, 64), dtype=np.int8) - 1)
        return np.cumprod(np.fliplr(self._challenge), axis=1, dtype=np.int8)[0]  # return self._challenge[0]

    def step(self, action):
        self._tries += 1
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self.puf.eval(self._challenge), [((2 * action) - 1)])
        reward = (1 / self._tries) if terminated else 0
        # return observation, reward, terminated, False, info
        return np.cumprod(np.fliplr(self._challenge), axis=1, dtype=np.int8)[
            0], reward, terminated, {}  # return self._challenge[0], reward, terminated, {}
