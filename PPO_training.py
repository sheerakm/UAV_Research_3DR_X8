import gymnasium as gym

from gymnasium_env.envs.UAV_SIM import UAVEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

# Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)

env = UAVEnv()
check_env(env, warn=True)

exit()


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("UAV_Training")


obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
