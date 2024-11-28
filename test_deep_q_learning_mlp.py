import time
import random
import numpy as np
from stable_baselines3 import DQN 
from stable_baselines3.common.vec_env import DummyVecEnv
from wrap import SnakeEnv

MODEL_PATH = r"models/deep_q_learning_mlp.zip"
NUM_EPISODES = 10
SILENT_MODE = False
FRAME_DELAY = 0.05
ROUND_DELAY = 1

def initialize_environment(seed, limit_step=True, silent_mode=False):
    env = SnakeEnv(seed=seed, limit_step=limit_step, silent_mode=silent_mode)
    return DummyVecEnv([lambda: env])

def print_episode_summary(episode, reward_sum, score, steps):
    print(f"Episode {episode + 1} | Reward: {reward_sum:.4f} | Score: {score} | Steps: {steps}")

def print_summary(total_score, min_score, max_score, total_reward, num_episodes):
    print(f"Average Score: {total_score / num_episodes:.2f} | Min Score: {min_score} | Max Score: {max_score} | Average Reward: {total_reward / num_episodes:.4f}")

def main():
    seed = random.randint(0, np.random.randint(1e9))
    print(f"Using seed = {seed} for testing.")
    env = initialize_environment(seed, limit_step=True, silent_mode=SILENT_MODE)

    model = DQN.load(MODEL_PATH, env=env)

    total_reward, total_score = 0, 0
    min_score, max_score = float('inf'), float('-inf')

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        episode_reward, done = 0, False
        num_steps = 0
        step_reward_accumulator = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            num_steps += 1

            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]

            if not SILENT_MODE:
                env.envs[0].render()
                time.sleep(FRAME_DELAY)
                
        episode_score = info[0]["snake_size"] - 3
        min_score = min(min_score, episode_score)
        max_score = max(max_score, episode_score)
        
        print_episode_summary(episode, episode_reward, episode_score, num_steps)
        total_reward += episode_reward
        total_score += episode_score

        if not SILENT_MODE:
            time.sleep(ROUND_DELAY)

    env.close()
    print_summary(total_score, min_score, max_score, total_reward, NUM_EPISODES)

if __name__ == "__main__":
    main()
