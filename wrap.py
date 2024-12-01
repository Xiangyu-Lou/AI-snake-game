import math
import gym
import numpy as np
from snake import SnakeGame

class SnakeEnv(gym.Env):
    def __init__(self, seed=522117, board_size=20, silent_mode=True, limit_step=True, use_cnn=False):
        super().__init__()
        self.use_cnn = use_cnn
        self._initialize_game(seed, board_size, silent_mode)
        self._initialize_spaces()
        self._initialize_environment_properties(limit_step)

    def reset(self):
        self.game.reset()
        self.done = False
        self.reward_step_counter = 0
        return self._generate_observation()

    def step(self, action):
        self.done, info = self.game.step(action)
        obs = self._generate_observation()
        reward = self._calculate_reward(info)
        return obs, reward, self.done, info

    def render(self):
        self.game.render()
        
    def render(self, record = False):
        self.game.render()
        if record:
            frame = self.game.get_frame()
            return frame


    def get_action_mask(self):
        action_mask = []
        for a in range(self.action_space.n):
            action_mask.append(self._check_action_validity(a))
        return np.array([action_mask])

    def _initialize_game(self, seed, board_size, silent_mode):
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()
        self.board_size = board_size
        self.grid_size = board_size ** 2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

    def _initialize_spaces(self):
        self.action_space = gym.spaces.Discrete(4)
        if self.use_cnn:
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(100, 100, 3),
                dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-1, high=1,
                shape=(self.game.board_size, self.game.board_size),
                dtype=np.float32
            )

    def _initialize_environment_properties(self, limit_step):
        self.done = False
        self.step_limit = self.grid_size if limit_step else 1e9
        self.reward_step_counter = 0

    def _calculate_reward(self, info):
        reward = 0.0
        self.reward_step_counter += 1

        if self.reward_step_counter > self.step_limit:
            self.reward_step_counter = 0
            self.done = True

        if self.done:
            self.step_limit = self.grid_size
            # reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth)
            reward = (info["snake_size"] - self.grid_size) * 0.1
        elif info["food_obtained"]:
            # reward = info["snake_size"] / self.grid_size
            reward = math.exp((self.grid_size - self.reward_step_counter) / self.grid_size)
            self.reward_step_counter = 0
            # self.step_limit = self.grid_size + (self.grid_size * 0.2 * info["snake_size"])
            self.step_limit = self.grid_size * math.exp(0.02 * info["snake_size"])
        else:
            reward = self._calculate_movement_reward(info)

        return reward * 0.1

    def _calculate_movement_reward(self, info):
        snake_head = info["snake_head_pos"]
        prev_snake_head = info["prev_snake_head_pos"]
        food_pos = info["food_pos"]

        if np.linalg.norm(snake_head - food_pos) < np.linalg.norm(prev_snake_head - food_pos):
            return 1 / info["snake_size"]
        else:
            return -1 / info["snake_size"]

    def _check_action_validity(self, action):
        row, col = self._calculate_next_position(action)
        return self._is_position_valid(row, col)

    def _calculate_next_position(self, action):
        row, col = self.game.snake[0]
        direction = self.game.direction

        if action == 0 and direction != "DOWN":  # UP
            row -= 1
        elif action == 1 and direction != "RIGHT":  # LEFT
            col -= 1
        elif action == 2 and direction != "LEFT":  # RIGHT
            col += 1
        elif action == 3 and direction != "UP":  # DOWN
            row += 1

        return row, col

    def _is_position_valid(self, row, col):
        snake_body = self.game.snake
        board_limits = (0 <= row < self.board_size) and (0 <= col < self.board_size)

        if (row, col) == self.game.food:
            body_collision = (row, col) in snake_body
        else:
            body_collision = (row, col) in snake_body[:-1]

        return board_limits and not body_collision

    def _generate_observation(self):
        if self.use_cnn:
            obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

            # Set the snake body to gray with linearly decreasing intensity from head to tail.
            obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)

            # Stack single layer into 3-channel-image.
            obs = np.stack((obs, obs, obs), axis=-1)

            # Set the snake head to green and the tail to blue
            obs[tuple(self.game.snake[0])] = [255, 255, 0]
            obs[tuple(self.game.snake[-1])] = [255, 100, 0]

            # Set the food to red
            obs[self.game.food] = [0, 0, 255]

            # Enlarge the observation to 84x84
            obs = np.repeat(np.repeat(obs, 5, axis=0), 5, axis=1)
        else:
            obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.float32)
            snake_positions = np.transpose(self.game.snake)
            obs[tuple(snake_positions)] = np.linspace(0.8, 0.2, len(self.game.snake), dtype=np.float32)
            obs[tuple(self.game.snake[0])] = 1.0
            obs[tuple(self.game.food)] = -1.0
        return obs