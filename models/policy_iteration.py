import math
import argparse
import numpy as np
import collections
from random import randint
from tqdm.auto import tqdm

from src.snake_game import SnakeGame

class PolicyIterationSnake:
    def __init__(self,
                 field_shape,
                 initial_games=30000,
                 test_games=1,
                 goal_steps=500,
                 lr=1e-2,
                 filename='snake_nn_2.tflearn',
                 visualize=True,
                 seed=1):

        self.width, self.height = field_shape
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.seed = seed
        self.visualize = visualize
        self.filename = filename
        self.vectors_and_keys = [
                [[-1,  0], 0],
                [[ 0,  1], 1],
                [[ 1,  0], 2],
                [[ 0, -1], 3]
                ]

        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)
        self.gamma = 0.95
        self.nu = 0.0005

        self.policy = collections.defaultdict(int)

    def _model_transits_rewards(self):
        for _ in tqdm(range(self.initial_games)):
            game = SnakeGame()
            _, prev_score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            prev_food_distance = self.get_food_distance(snake, food)
            for _ in range(self.goal_steps):
                action, game_action = self.generate_action(snake)
                done, score, snake, food  = game.step(game_action)
                if done:
                    cur_observation = (-1, 0, 0, 0)
                    self.rewards[(prev_observation, action, cur_observation)] = -100
                    self.transits[(prev_observation, action)][cur_observation] += 1
                    break
                else:
                    cur_observation = self.generate_observation(snake, food)
                    food_distance = self.get_food_distance(snake, food)
                    if score > prev_score:
                        self.rewards[(prev_observation, action, cur_observation)] = 500
                    elif food_distance < prev_food_distance:
                        self.rewards[(prev_observation, action, cur_observation)] = 0
                    else:
                        self.rewards[(prev_observation, action, cur_observation)] = -10
                    self.transits[(prev_observation, action)][cur_observation] += 1
                    prev_observation = cur_observation
                    prev_food_distance = food_distance

    def generate_action(self, snake):
        action = randint(0, 2) - 1
        return action, self.get_game_action(snake, action)

    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)
        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
        return game_action

    def generate_observation(self, snake, food):
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, food_direction)
        quater = self.get_quater(angle)
        return (int(barrier_left), int(barrier_front), int(barrier_right), quater)

    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[0])

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[1:] \
               or point[0] == -1 \
               or point[1] == -1 \
               or point[0] == self.width \
               or point[1] == self.height

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def get_quater(self, a):
        if a < -0.5:
            return 0
        elif a < 0:
            return 1
        elif a == 0:
            return 2
        elif a < 0.5:
            return 3
        else:
            return 4

    def state_to_int(self, state):
        return (state[0] * 4 + state[1] * 2 + state[2]) * 8 + state[3]

    def _get_action_value(self, observation, action):
        
        next_observation_counts = self.transits[(observation, action)]
        total_transits = sum(next_observation_counts.values())
        action_value = 0.0
        
        for next_observation, n_transits in next_observation_counts.items():
            reward = self.rewards[(observation, action, next_observation)]
            transit_prob = (n_transits / total_transits)
            action_value += transit_prob * (reward + self.gamma * self.values[next_observation])
        
        return action_value

    def _get_best_action(self, observation):
        
        action_values = {}

        for action in range(-1, 2):
            action_value = self._get_action_value(observation, action)
            action_values[action] = action_value

        best_action_value = max(action_values.values())
        best_action = max(action_values, key=action_values.get)
        
        return best_action

    def _policy_iteration(self):

        delta = 0
        iter_count = 1

        while True:
            delta = 0
            state = (-1, 0, 0, 0)
            v = self.values[state]
            action = self.policy[state]
            self.values[state] = self._get_action_value(state, action)
            delta = max(delta, abs(v - self.values[state]))
            for a in (0, 1):
                for b in (0, 1):
                    for c in (0, 1):
                        for d in range(5):
                            state = (a, b, c, d)
                            v = self.values[state]
                            action = self.policy[state]
                            self.values[state] = self._get_action_value(state, action)
                            delta = max(delta, abs(v - self.values[state]))
            
            iter_count += 1

            if delta < self.nu:
                break

        policy_stable = np.ones((2 ** 6), dtype=bool)

        state = (-1, 0, 0, 0)
        old_policy = self.policy[state]
        self.policy[state] = self._get_best_action(state)
        policy_stable[self.state_to_int(state)] = old_policy == self.policy[state]
        for a in (0, 1):
                for b in (0, 1):
                    for c in (0, 1):
                        for d in range(5):
                            state = (a, b, c, d)
                            old_policy = self.policy[state]
                            self.policy[state] = self._get_best_action(state)
                            policy_stable[self.state_to_int(state)] = old_policy == self.policy[state]
        
        return policy_stable
    
    def run(self):
        self._model_transits_rewards()
        while True:
            policy_stable = self._policy_iteration()
            if policy_stable.all() == True:
                break

        for _ in range(self.test_games):
            game = SnakeGame(board_width=self.width,
                             board_height=self.height,
                             gui=self.visualize,
                             seed=self.seed,
                             title='Policy Iteration')

            _, _, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            
            for _ in range(self.goal_steps):
                action = self.policy[prev_observation]
                game_action = self.get_game_action(snake, action)
                done, _, snake, food = game.step(game_action)
                if done:
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy Iteration")

    parser.add_argument('--visualize', type=bool, default=True,
                        help='Visualize Snake Game')

    parser.add_argument('--field-shape', type=tuple, default=(12, 12),
                        help='Field shape')

    args = parser.parse_args()
    visualize = args.visualize
    field_shape = args.field_shape

    PolicyIterationSnake(field_shape=field_shape).run()