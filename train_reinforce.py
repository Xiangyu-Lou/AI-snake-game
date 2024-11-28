import os
import time
import random
import numpy as np
import tensorflow as tf
from wrap_2 import SnakeEnv

# 设置模型保存路径
MODEL_DIR = "models/"
MODEL_PATH = os.path.join(MODEL_DIR, "reinforce")
os.makedirs(MODEL_DIR, exist_ok=True)

# 超参数配置
TOTAL_EPISODES = 5000
GAMMA = 0.99
LEARNING_RATE = 0.001
RENDER = True
FRAME_DELAY = 0.01  # 渲染延迟
ROUND_DELAY = 1  # 回合间延迟

# 初始化环境
seed = random.randint(0, int(1e9))
print(f"Using seed = {seed} for training.")
env = SnakeEnv(seed=seed, limit_step=True, silent_mode=not RENDER)

# 策略网络（CNN）
class PolicyNetwork(tf.keras.Model):
    def __init__(self, action_dim, input_shape):
        super(PolicyNetwork, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='softmax')  # 动作概率分布

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.output_layer(x)

# 折扣奖励计算
def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    cumulative_reward = 0
    for t in reversed(range(len(rewards))):
        cumulative_reward = rewards[t] + gamma * cumulative_reward
        discounted_rewards[t] = cumulative_reward
    return discounted_rewards

# 初始化网络和优化器
input_shape = env.observation_space.shape
action_dim = env.action_space.n
policy_network = PolicyNetwork(action_dim, input_shape)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# 保存模型
def save_model(model, model_path):
    model.save_weights(model_path)
    print(f"Model saved to {model_path}")

# 训练主逻辑
for episode in range(TOTAL_EPISODES):
    state = env.reset()
    episode_states, episode_actions, episode_rewards = [], [], []
    done = False
    num_steps = 0

    while not done:
        # 转换状态为 TensorFlow 格式
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = policy_network(state)
        action = np.random.choice(env.action_space.n, p=action_probs.numpy()[0])

        next_state, reward, done, info = env.step(action)

        # 保存数据
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)
        num_steps += 1

        state = next_state

        # 渲染环境
        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)

    # 计算折扣奖励
    discounted_rewards = compute_discounted_rewards(episode_rewards, GAMMA)
    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

    # 计算损失并更新网络
    with tf.GradientTape() as tape:
        total_loss = 0
        for state, action, reward in zip(episode_states, episode_actions, discounted_rewards):
            action_probs = policy_network(state)
            log_prob = tf.math.log(action_probs[0, action])
            total_loss -= log_prob * reward  # REINFORCE 损失
    grads = tape.gradient(total_loss, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))

    # 打印训练日志
    print(f"Episode {episode + 1}/{TOTAL_EPISODES} | Total Reward: {sum(episode_rewards):.2f} | Steps: {num_steps} | Loss: {total_loss.numpy():.4f}")

    # 回合间延迟
    if RENDER:
        time.sleep(ROUND_DELAY)

# 保存模型
save_model(policy_network, MODEL_PATH)
env.close()
