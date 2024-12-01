import os
import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from wrap import SnakeEnv
import time

NUM_ENVS = 32
LOG_DIR = "logs"
MODEL_DIR = "models"
TOTAL_TIMESTEPS = 20_000_000 #200_000_000 300_000_000

def create_action_mask(env):
    return env.get_action_mask()

def initialize_environment(seed):
    def _init():
        env = SnakeEnv(seed=seed, use_cnn=True)
        return ActionMasker(env, create_action_mask)
    return _init

def create_vectorized_environments(num_envs):
    return SubprocVecEnv([
        initialize_environment(seed=np.random.randint(0, 1e9)) for _ in range(num_envs)
    ])
    
# linear schedule for learning rate
def linear_schedule(initial_value, final_value):
    def func(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return func

def configure_dqn_model(envs):
    return DQN(
        "CnnPolicy",
        envs,
        learning_rate=linear_schedule(2.5e-4, 2.5e-6), # 1e-4, 1e-6
        buffer_size=10000,
        learning_starts=1000, # 5000
        batch_size=512,
        tau=0.1,
        gamma=0.98, # 0.96
        train_freq=8, # 4
        target_update_interval=1000, # 5000
        exploration_fraction=0.75, # 0.8 0.9
        exploration_final_eps=0.01, # 0.01 0.005
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda"
    )

def train_model(model, total_timesteps, model_dir, model_name="deep_q_learning_cnn"):
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(model_dir, model_name))

def main():
    envs = create_vectorized_environments(NUM_ENVS)
    model = configure_dqn_model(envs)
    time_start = time.time()
    train_model(model, TOTAL_TIMESTEPS, MODEL_DIR)
    time_use = time.time() - time_start
    print(f"Training time: {(time_use / 3600 ):.2f} hours")
    envs.close()

if __name__ == "__main__":
    main()
