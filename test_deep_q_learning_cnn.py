import time
import random
import numpy as np
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from wrap import SnakeEnv
import moviepy.editor as mpy

MODEL_PATH = r"models/backup/deep_q_learning_cnn.zip"
NUM_EPISODES = 50
SILENT_MODE = False
FRAME_DELAY = 0.002
ROUND_DELAY = 0
VIDEO_OUTPUT_DIR = "videos"
RECORD_VIDEO = False

def initialize_environment(seed, limit_step=True, silent_mode=False):
    env = SnakeEnv(seed=seed, limit_step=limit_step, silent_mode=silent_mode, use_cnn=True)
    return DummyVecEnv([lambda: env])

def print_episode_summary(episode, reward_sum, score, steps):
    print(f"Episode {episode + 1} | Reward: {reward_sum:.4f} | Score: {score} | Steps: {steps}")

def print_summary(total_score, min_score, max_score, total_reward, num_episodes):
    print(f"Average Score: {total_score / num_episodes:.2f} | Min Score: {min_score} | Max Score: {max_score} | Average Reward: {total_reward / num_episodes:.4f}")

def save_video(frames, filename, fps=30):
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(os.path.join(VIDEO_OUTPUT_DIR, filename), codec="libx264")
    print(f"Saved video: {filename}")

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

        frames = []  # save frames for video

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            num_steps += 1
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]

            # render the game and record the frames
            if not SILENT_MODE:
                frame = env.envs[0].render(record=True)
                if RECORD_VIDEO:
                    frames.append(frame)
                time.sleep(FRAME_DELAY)

        episode_score = info[0]["snake_size"] - 3
        min_score = min(min_score, episode_score)
        max_score = max(max_score, episode_score)
        
        # only save video for episodes with score greater than 58
        if RECORD_VIDEO and episode_score > 58:
            video_filename = f"{episode_score}.mp4"
            save_video(frames, video_filename)

        print_episode_summary(episode, episode_reward, episode_score, num_steps)

        total_reward += episode_reward
        total_score += episode_score

        if not SILENT_MODE:
            time.sleep(ROUND_DELAY)

    env.close()

    print_summary(total_score, min_score, max_score, total_reward, NUM_EPISODES)

if __name__ == "__main__":
    main()