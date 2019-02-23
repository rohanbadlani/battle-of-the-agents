import os
import sys
import gym
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

import numpy as np
import imageio

def evaluate(env, model, eval_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param eval_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward
    """
    episode_rewards = [[0.0] for _ in range(env.num_envs)]
    obs = env.reset()
    img = env.render(mode='rgb_array')
    images = []                                                                                              

    for j in range(eval_steps):
      images.append(img)
      # _states are only useful when using LSTM policies
      actions, _states = model.predict(obs)
      # here, action, rewards and dones are arrays
      # because we are using vectorized env
      obs, rewards, dones, info = env.step(actions)
      
      img = env.render(mode='rgb_array')

      # Stats
      for i in range(env.num_envs):
          episode_rewards[i][-1] += rewards[i]
          if dones[i]:
              episode_rewards[i].append(0.0)

    mean_rewards =  [0.0 for _ in range(env.num_envs)]
    n_episodes = 0
    for i in range(env.num_envs):
        mean_rewards[i] = np.mean(episode_rewards[i])     
        n_episodes += len(episode_rewards[i])   

    # Compute mean reward
    mean_reward = round(np.mean(mean_rewards), 1)
    print("Mean reward:", mean_reward, "Num episodes:", n_episodes)

    return mean_reward, n_episodes, images

# Load the agent and evaluate
def createGif(images, gif_dir):
    imageio.mimsave(gif_dir + "render.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)


if(len(sys.argv)!=6):
    print("evaluate_dqn usage: Please provide the following arguments in order. \n\n 1. env_name (LunarLander-v2, Doom, Fetch) \n 2. train_timesteps (int) - The timesteps you trained on. \n 3. evaluation_timesteps (int) - The timesteps you want to test on. \n 4. model_dir (str)\n 5. results_dir (str)\n")
    sys.exit(1)

env_name = str(sys.argv[1])
train_timesteps = int(sys.argv[2])
evaluate_timesteps = int(sys.argv[3])
model_dir = str(sys.argv[4])
results_dir = str(sys.argv[5])

if not model_dir.endswith("/"):
    model_dir += "/"

os.makedirs(results_dir, exist_ok=True)
if not results_dir.endswith("/"):
    results_dir += "/"

env = gym.make(env_name)
env = DummyVecEnv([lambda: env])

model = DQN.load(str(model_dir) + "dqn_" + str(env_name) + "_trained_timesteps_" + str(train_timesteps))

mean_rewards, eval_episodes, images = evaluate(env, model, evaluate_timesteps)
createGif(images, results_dir)

with open(results_dir + "results.txt", "w") as fp:
    fp.write("Env\tTrain-Timesteps\tTest-Timesteps\tTest-Episodes\tAvg Reward\n")
    fp.write(str(env_name) + "\t" + str(train_timesteps) + "\t" + str(evaluate_timesteps) + "\t" + str(eval_episodes) + "\t" + str(mean_rewards) + "\n")
fp.close()