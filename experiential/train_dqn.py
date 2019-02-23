import os
import gym
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

# Create and wrap the environment
env = gym.make('LunarLander-v2')
env = DummyVecEnv([lambda: env])

model = DQN(MlpPolicy, env, verbose=1)

# Train the agent
model.learn(total_timesteps=2500)

# Save the agent
model.save("dqn_lunar")
del model  # delete trained model