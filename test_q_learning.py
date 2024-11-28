import time
import random
import numpy as np

from wrap import SnakeEnv

MODEL_PATH = r"models/q_learning.npy"

NUM_EPISODE = 10

RENDER = True
FRAME_DELAY = 0.05 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

if RENDER:
    env = SnakeEnv(seed=seed, limit_step=False, silent_mode=False)
else:
    env = SnakeEnv(seed=seed, limit_step=False, silent_mode=True)

# Load the trained Q-table
q_table = np.load(MODEL_PATH)

def state_to_index(state, env):
    # use a hash function to convert the state to an index(avoide the state is not exist in the q_table)
    return hash(state.tostring()) % (env.observation_space.shape[0] * env.observation_space.shape[1])

total_reward = 0
total_score = 0
min_score = 1e9
max_score = 0

for episode in range(NUM_EPISODE):
    obs = env.reset()
    obs = obs.flatten()
    episode_reward = 0
    done = False
    
    num_step = 0
    info = None

    sum_step_reward = 0

    print(f"=================== Episode {episode + 1} ==================")

    while not done:
        state_index = state_to_index(obs, env)
        action = np.argmax(q_table[state_index])
        
        num_step += 1

        obs, reward, done, info = env.step(action)
        obs = obs.flatten()
        
        if done:
            last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
            print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")
        
        elif info["food_obtained"]:
            print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
            sum_step_reward = 0 # Reset step reward accumulator.

        else:
            sum_step_reward += reward # Accumulate step rewards.
            
        episode_reward += reward

        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)

    episode_score = env.game.score
    if episode_score < min_score:
        min_score = episode_score
    if episode_score > max_score:
        max_score = episode_score
    
    snake_size = info["snake_size"] + 1
    print(f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, Total Steps: {num_step}, Snake Size: {snake_size}")
    total_reward += episode_reward
    total_score += env.game.score
    if RENDER:
        time.sleep(ROUND_DELAY)

env.close()
print(f"=================== Summary ==================")
print(f"Average Score: {total_score / NUM_EPISODE}, Min Score: {min_score}, Max Score: {max_score}, Average reward: {total_reward / NUM_EPISODE}")