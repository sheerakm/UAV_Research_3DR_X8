import gymnasium as gym

from gymnasium_env.envs.UAV_SIM import UAVEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

# Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)

env = UAVEnv()
check_env(env, warn=True)


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
# model.save("UAV_Training")


obs, info = env.reset()
print("outside loop")
print(obs)
while True:
    print("starting anew ")

    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    print(obs)

    if terminated or truncated :
        obs = env.reset()
        print("finished")
        print("starting anew ")
        print(obs)
    # env.render()
