import time
import random
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from wrap_2 import SnakeEnv

# 模型路径和测试参数
MODEL_PATH = r"models/deep_q_learning_cnn.zip"
NUM_EPISODES = 100
SILENT_MODE = False
FRAME_DELAY = 0.005
ROUND_DELAY = 1

# 初始化环境
def initialize_environment(seed, limit_step=True, silent_mode=False):
    env = SnakeEnv(seed=seed, limit_step=limit_step, silent_mode=silent_mode, use_cnn=True)
    return DummyVecEnv([lambda: env])

# 打印每一回合的总结
def print_episode_summary(episode, reward_sum, score, steps):
    print(f"Episode {episode + 1} | Reward: {reward_sum:.4f} | Score: {score} | Steps: {steps}")

# 打印所有回合的总结
def print_summary(total_score, min_score, max_score, total_reward, num_episodes):
    print(f"Average Score: {total_score / num_episodes:.2f} | Min Score: {min_score} | Max Score: {max_score} | Average Reward: {total_reward / num_episodes:.4f}")

# 主函数
def main():
    # 随机种子初始化
    seed = random.randint(0, np.random.randint(1e9))
    print(f"Using seed = {seed} for testing.")

    # 初始化环境
    env = initialize_environment(seed, limit_step=True, silent_mode=SILENT_MODE)

    # 加载训练好的 CNN 模型
    model = DQN.load(MODEL_PATH, env=env)

    # 初始化统计变量
    total_reward, total_score = 0, 0
    min_score, max_score = float('inf'), float('-inf')

    # 开始测试
    for episode in range(NUM_EPISODES):
        obs = env.reset()
        episode_reward, done = 0, False
        num_steps = 0

        while not done:
            # 模型预测动作（确定性）
            action, _ = model.predict(obs, deterministic=True)
            num_steps += 1

            # 执行动作
            obs, reward, done, info = env.step(action)

            # 累积奖励
            episode_reward += reward[0]

            # 渲染环境
            if not SILENT_MODE:
                env.envs[0].render()
                time.sleep(FRAME_DELAY)

        # 记录当前回合得分（蛇长度-初始长度）
        episode_score = info[0]["snake_size"] - 3
        min_score = min(min_score, episode_score)
        max_score = max(max_score, episode_score)

        # 打印回合结果
        print_episode_summary(episode, episode_reward, episode_score, num_steps)

        # 更新统计变量
        total_reward += episode_reward
        total_score += episode_score

        # 控制每回合的间隔
        if not SILENT_MODE:
            time.sleep(ROUND_DELAY)

    # 关闭环境
    env.close()

    # 打印总结
    print_summary(total_score, min_score, max_score, total_reward, NUM_EPISODES)

if __name__ == "__main__":
    main()
