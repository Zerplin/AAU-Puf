import logging
import os
import sys

import chainer
import chainerrl
from chainerrl import policies
from chainerrl.agents import a3c
from chainerrl.agents.a3c import A3C, A3CModel
import gym
import numpy as np
from chainerrl import links

# noinspection PyUnresolvedReferences
import gym_env

# *** Settings ***
challenge_bit_length = 64
arbiter_seed = 1337
M_delay_granularity = 2
evaluation_interval = 10 ** 6

outdir = 'result_A3C_x' + str(challenge_bit_length) + '_M' + str(M_delay_granularity)
if not os.path.exists(outdir):
    os.makedirs(outdir)

env = gym.make('gym_env/ArbiterPufDelayII-v0', challenge_bit_length=challenge_bit_length, arbiter_seed=arbiter_seed,
               M_delay_granularity=M_delay_granularity)
print('observation space:', env.observation_space)
print('action space:', env.action_space)
obs = env.reset()

# Set up the logger to print info messages for understandability.
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')


class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


model = A3CFFSoftmax(env.observation_space.shape[0], env.action_space.n)
# GPU?
#model.to_gpu(0)
optimizer = chainer.optimizers.Adam(eps=1e-3)
optimizer.setup(model)
agent = A3C(model, optimizer, t_max=5, gamma=0.99, beta=0.1, phi=lambda x: x.astype(np.float32, copy=False))

chainerrl.experiments.train_agent_with_evaluation(agent, env, steps=100000000,  # Train the agent for 10 mio steps
                                                  eval_n_steps=None,  # We evaluate for episodes, not time
                                                  eval_n_episodes=10000,  # 10000 challenges sampled for each evaluation
                                                  eval_interval=evaluation_interval,  # Evaluate the agent after x steps
                                                  successful_score=0.99,  # early stopping if mean is > 99%
                                                  outdir=outdir)  # Save everything to 'result' directory
