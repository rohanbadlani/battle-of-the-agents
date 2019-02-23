import os
import sys
import gym
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor

import pdb

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True

def train(log_dir, model_dir, env_name, train_timesteps=2500):
    #make sure dir exists
    os.makedirs(log_dir, exist_ok=True)

    os.makedirs(model_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make(env_name)

    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = DQN(MlpPolicy, env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=train_timesteps)
    
    # Save the agent
    if not model_dir.endswith("/"):
        model_dir += "/"

    model.save(str(model_dir) + "dqn_" + str(env_name) + "_trained_timesteps_" + str(train_timesteps))
    # delete trained model
    del model


def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, plot_dir, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    os.makedirs(plot_dir, exist_ok=True)

    pdb.set_trace()
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = movingAverage(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")

    if not plot_dir.endswith("/"):
        plot_dir += "/"

    plt.savefig(str(plot_dir) + "results_dqn_" + str(env_name) + "_trained_timesteps_" + str(train_timesteps))

if(len(sys.argv)!=6):
    print("train_dqn usage: Please provide the following arguments in order. \n\n 1. env_name (LunarLander-v2, Doom, Fetch) \n 2. train_timesteps (int) \n 3. log_folder (str) \n 4. model_dir (str)\n 5. plot_dir (str) \n")
    sys.exit(1)

env_name = str(sys.argv[1])
train_timesteps = int(sys.argv[2])
log_dir = str(sys.argv[3])
model_dir = str(sys.argv[4])
plot_dir = str(sys.argv[5])

train(log_dir, model_dir, env_name, train_timesteps)
#plot_results(log_dir, plot_dir)