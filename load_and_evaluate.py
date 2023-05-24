import gym
import numpy as np
from chainer import optimizers
from chainerrl import agents, explorers, q_functions, replay_buffer

# noinspection PyUnresolvedReferences
import gym_env

# Define your environment
env = gym.make('gym_env/PufAttack-v0')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# Define Q-Function
q_func = q_functions.FCStateQFunctionWithDiscreteAction(
    obs_size, n_actions,
    n_hidden_layers=2, n_hidden_channels=64)

# Specify optimizer
optimizer = optimizers.Adam(eps=1e-3)
optimizer.setup(q_func)

# Define replay buffer
rbuf = replay_buffer.ReplayBuffer(10 ** 3)

# Specify exploration strategy
explorer = explorers.ConstantEpsilonGreedy(
    epsilon=0.01, random_action_func=env.action_space.sample)

# Define agent
phi = lambda x: x.astype(np.float32, copy=False)
agent = agents.DoubleDQN(
    q_func, optimizer, rbuf, gpu=-1, gamma=0.99,
    explorer=explorer, replay_start_size=500,
    target_update_interval=100,
    update_interval=1, phi=phi, target_update_method='soft')

# Load agent from a saved file
agent.load('result/11000_finish')

# Now you can evaluate the agent in the environment
count = 0
n_episodes = 100000
for i in range(n_episodes):
    obs = env.reset()
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while not done and t < 2:
        # Uncomment to watch the policy in action
        # env.render()
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
    count += reward
    print('Episode finished after {} timesteps, total rewards {}'.format(t, R))
env.close()
print(f"{(count / n_episodes):.4f}")
