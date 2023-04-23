from gym.envs.registration import register

register(id='gym_env/PufAttack-v0', entry_point='gym_env.envs:PufAttackEnv', max_episode_steps=1, )
