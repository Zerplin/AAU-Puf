from gym.envs.registration import register

register(id='gym_env/PufAttack-v0', entry_point='gym_env.envs:PufAttackEnv0', max_episode_steps=1, )
register(id='gym_env/PufAttack-v1', entry_point='gym_env.envs:PufAttackEnv1', max_episode_steps=2, )
register(id='gym_env/ArbiterPufDelayII-v0', entry_point='gym_env.envs:ArbiterPufDelayII', max_episode_steps=2, )
