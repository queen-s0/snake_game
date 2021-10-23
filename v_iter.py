from snake_game import SnakeGame
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter
import collections
from tqdm.auto import tqdm

class SnakeNN:
    def __init__(self, initial_games = 30000, test_games = 10, goal_steps = 500, lr = 1e-2, filename = 'snake_nn_2.tflearn'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.vectors_and_keys = [
                [[-1, 0], 0],
                [[0, 1], 1],
                [[1, 0], 2],
                [[0, -1], 3]
                ]

    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            game = SnakeGame()
            _, prev_score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            prev_food_distance = self.get_food_distance(snake, food)
            for _ in range(self.goal_steps):
                action, game_action = self.generate_action(snake)
                done, score, snake, food  = game.step(game_action)
                if done:
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    break
                else:
                    food_distance = self.get_food_distance(snake, food)
                    if score > prev_score or food_distance < prev_food_distance:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(snake, food)
                    prev_food_distance = food_distance
        return training_data

    def generate_action(self, snake):
        action = randint(0,2) - 1
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
        food_dist = abs(food_direction[0]) + abs(food_direction[1])
        return (int(barrier_left), int(barrier_front), int(barrier_right), food_dist)

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

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
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def get_quater(self, v):
        a, b = v > 0
        if a and not b:
            return 0
        elif not a and not b:
            return 1
        elif not a and b:
            return 2
        else:
            return 3

    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X,y, n_epoch = 3, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = SnakeGame()
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                   predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                done, score, snake, food  = game.step(game_action)
                game_memory.append([prev_observation, action])
                if done:
                    print('-----')
                    print(steps)
                    print(snake)
                    print(food)
                    print(prev_observation)
                    print(predictions)
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:',mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:',mean(scores_arr))
        print(Counter(scores_arr))

    def visualise_game(self, model):
        game = SnakeGame(gui = True)
        _, _, snake, food = game.start()
        prev_observation = self.generate_observation(snake, food)
        for _ in range(self.goal_steps):
            precictions = []
            for action in range(-1, 2):
               precictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
            action = np.argmax(np.array(precictions))
            game_action = self.get_game_action(snake, action - 1)
            done, _, snake, food  = game.step(game_action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(snake, food)

    def train(self):
        training_data = self.initial_population()
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model)

    def visualise(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)


class ValueIteration(SnakeNN):
    def __init__(self, initial_games = 10000, test_games = 10, goal_steps = 500):
        super().__init__(initial_games, test_games, goal_steps)
        
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)
        self.state_n = 8 * 38
        self.gamma = 0.95
        self.nu = 0.0005

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
                    else:
                        self.rewards[(prev_observation, action, cur_observation)] = -10
                    self.transits[(prev_observation, action)][cur_observation] += 1
                    prev_observation = cur_observation
                    prev_food_distance = food_distance

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

        for action in range(3):
            action_value = self._get_action_value(observation, action)
            action_values[action] = action_value

        best_action_value = max(action_values.values())
        best_action = max(action_values, key=action_values.get)
        
        return best_action

    def _value_iteration(self):
        delta = 0
        iter_count = 1

        while True:
            delta = 0
            
            state = (-1, 0, 0, 0)
            v = self.values[state]
            u = self._get_best_action(state)
            self.values[state] = self._get_action_value(state, u)
            delta = max(delta, abs(v - self.values[state]))
            for a in (0, 1):
                for b in (0, 1):
                    for c in (0, 1):
                        for d in range(38):
                            state = (a, b, c, d)
                            v = self.values[state]
                            u = self._get_best_action(state)
                            self.values[state] = self._get_action_value(state, u)
                            delta = max(delta, abs(v - self.values[state]))
            
            iter_count += 1

            if delta < self.nu:
                break                   

    def run(self):
        self._model_transits_rewards()
        self._value_iteration()
        print(self.transits)
        print()
        print(self.rewards)
        assert False

        for _ in range(self.test_games):
            game = SnakeGame(gui = True)
            _, _, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                action = self._get_best_action(prev_observation)
                game_action = self.get_game_action(snake, action - 1)
                done, _, snake, food  = game.step(game_action)
                if done:
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)

    
if __name__ == "__main__":
    ValueIteration().run()