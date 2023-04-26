import chainer
import chainerrl
import gym
import numpy as np

import gym_env

env = gym.make('gym_env/PufAttack-v0')
print('observation space:', env.observation_space)
print('action space:', env.action_space)

obs = env.reset()
print('initial observation:', obs)

action = env.action_space.sample()
obs, r, done, info = env.step(action)
print('next observation:', obs)
print('reward:', r)
print('done:', done)
print('info:', info)

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
print('observation size:', obs_size)
print('num of actions:', n_actions)

q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(obs_size, n_actions, n_hidden_layers=2,
    n_hidden_channels=64)

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

# Set the discount factor that discounts future rewards.
gamma = 0.95

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
agent = chainerrl.agents.DoubleDQN(q_func, optimizer, replay_buffer, gamma, explorer, replay_start_size=500,
                                   update_interval=1, target_update_interval=100, phi=phi)

# Set up the logger to print info messages for understandability.
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

chainerrl.experiments.train_agent_with_evaluation(agent, env, steps=2000,  # Train the agent for 2000 steps
                                                  eval_n_steps=None,  # We evaluate for episodes, not time
                                                  eval_n_episodes=10,  # 10 episodes are sampled for each evaluation
                                                  train_max_episode_len=200,  # Maximum length of each episode
                                                  eval_interval=1000,  # Evaluate the agent after every 1000 steps
                                                  outdir='result')  # Save everything to 'result' directory
