import gym
import numpy as np
import pandas as pd
from gym import spaces


class PufAttackEnv(gym.Env):
    def __init__(self, render_mode=None, data='challenge_response_100k.csv'):
        df = pd.read_csv(data)
        print(df.shape)

        for col in df.columns.values:
            df[col] = df[col].astype('int64')

        df = df.replace([-1], 0)
        print(df.head())
        print(df.describe())
        print(df['64'].value_counts())

        self._df = df
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Dict({"agent": spaces.MultiBinary(64), "target": spaces.MultiBinary(64), })
        self.observation_space = spaces.MultiBinary(64)

        self.action_space = spaces.Discrete(2)
        self.render_mode = None
        self.viewer = None
        self.window = None
        self.clock = None

    def _get_obs(self):
        # return {"agent": self._challenge, "target": self._challenge}
        return self._challenge

    # def _get_info(self):
    #     return {"distance": np.linalg.norm(self._response - self._previous_try, ord=1)}

    def reset(self, seed=None, options=None):
        # Choose the agent's location uniformly at random
        df = self._df.sample()

        self._challenge = df.drop('64', axis=1).to_numpy().astype(np.int8)[0]

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._response = df['64'].to_numpy().astype(np.int8)
        self._previous_try = df['64'].replace([0, 1], [1, 0]).to_numpy().astype(np.int8)

        observation = self._get_obs()
        # info = self._get_info()

        # return observation, info
        return observation

    def step(self, action):
        # An episode is done iff the agent has reached the target
        self._previous_try = [action]
        terminated = np.array_equal(self._response, [action])
        reward = 1 if terminated else 0  # Binary sparse rewards
        # return observation, reward, terminated, False, info
        return self._get_obs(), reward, terminated, {}
