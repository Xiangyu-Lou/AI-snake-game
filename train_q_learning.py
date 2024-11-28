import os
import sys
import random
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from wrap import SnakeEnv

NUM_EPISODES = 100000
MAX_STEPS = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def initialize_q_table(env):
    state_space_size = (env.observation_space.shape[0] * env.observation_space.shape[1],)
    action_space_size = env.action_space.n
    q_table = np.zeros(state_space_size + (action_space_size,))
    return q_table

def state_to_index(state, env):
    return hash(state.tostring()) % (env.observation_space.shape[0] * env.observation_space.shape[1])

def choose_action(state, q_table, epsilon, env):
    state_index = state_to_index(state, env)
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state_index])

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma, env):
    state_index = state_to_index(state, env)
    next_state_index = state_to_index(next_state, env)
    best_next_action = np.argmax(q_table[next_state_index])
    td_target = reward + gamma * q_table[next_state_index][best_next_action]
    td_error = td_target - q_table[state_index][action]
    q_table[state_index][action] += alpha * td_error

def main():
    env = SnakeEnv()
    q_table = initialize_q_table(env)
    global EPSILON

    for episode in range(NUM_EPISODES):
        seed = random.randint(0, 1e9)
        env.seed(seed)
        state = env.reset()
        state = state.flatten()
        total_reward = 0

        for step in range(MAX_STEPS):
            action = choose_action(state, q_table, EPSILON, env)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            update_q_table(q_table, state, action, reward, next_state, LEARNING_RATE, DISCOUNT_FACTOR, env)
            state = next_state
            total_reward += reward

            if done:
                break

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

        if episode % 100 == 0:
            with open(os.path.join(LOG_DIR, "q_learning_log.txt"), "a") as f:
                f.write(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {EPSILON}, Seed: {env.seed}\n")
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {EPSILON}, Seed: {seed}")

    env.close()

    np.save(os.path.join(MODEL_DIR, "q_learning.npy"), q_table)

if __name__ == "__main__":
    main()