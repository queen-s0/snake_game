import math
import tflearn
import argparse
import numpy as np
from random import randint
from statistics import mean
from collections import Counter

from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, fully_connected

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.snake_game import SnakeGame

class CustomDataset(Dataset):

    def __init__(self, X, y):
        super(CustomDataset, self).__init__()

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(5, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class SnakeNN:
    def __init__(self,
                 field_shape,
                 initial_games=10000,
                 test_games=1,
                 goal_steps=2000,
                 lr=1e-2,
                 filename='',
                 seed=52,
                 visualize=True,
                 network_type='tf'):

        self.width, self.height = field_shape
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.seed = seed
        self.network_type = network_type
        self.visualize = visualize
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
        angle = self.get_angle(snake_direction, food_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])

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
        return point.tolist() in snake[1:] \
               or point[0] == 0 \
               or point[1] == 0 \
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

        if self.network_type == 'tf':

            model.fit(X,y, n_epoch = 2, shuffle = True, run_id = self.filename)
            model.save(self.filename)

            return model

        elif self.network_type == 'torch':

            model_torch = MLP()
            lr = 1e-2
            n_epochs = 3

            optimizer = torch.optim.Adam(model_torch.parameters(), lr=lr)
            criterion = nn.MSELoss()

            X = torch.squeeze(torch.from_numpy(X)).type(torch.float32)
            y = torch.squeeze(torch.from_numpy(y)).type(torch.float32)

            dataset = CustomDataset(X, y)
            loader = DataLoader(dataset, shuffle=True, batch_size=50)

            print(X.shape, y.shape)

            for epoch in range(n_epochs):
                for index, batch in enumerate(loader):
                    X_sample, y_sample = batch

                    optimizer.zero_grad()
                    logits = model_torch(X_sample)

                    loss = criterion(logits, y_sample)

                    loss.backward()
                    optimizer.step()

                print('Epoch', epoch, 'loss', loss.item())

            torch.save(model_torch, 'torch_model')
            return model_torch

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = SnakeGame(board_width=self.width, board_height=self.height, gui=self.visualize)
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                    if self.network_type == 'torch':
                        input = torch.from_numpy(
                            self.add_action_to_observation(prev_observation, action).reshape(-1, 5)).type(torch.float32)
                        prediction = model(input).detach().numpy()

                    elif self.network_type == 'tf':
                        prediction = model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1))

                    predictions.append(prediction)

                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                done, score, snake, food = game.step(game_action)
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
        print('Average steps:', mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:', mean(scores_arr))
        print(Counter(scores_arr))

    def visualise_game(self, model):
        game = SnakeGame(board_width=self.width,
                         board_height=self.height,
                         gui=self.visualize,
                         title='MLP')
        _, _, snake, food = game.start()
        prev_observation = self.generate_observation(snake, food)
        for _ in range(self.goal_steps):
            precictions = []
            for action in range(-1, 2):
                if network_type == 'torch':
                    input = torch.from_numpy(
                        self.add_action_to_observation(prev_observation, action).reshape(-1, 5)).type(
                        torch.float32)
                    prediction = model(input).detach().numpy()

                elif network_type == 'tf':
                    prediction = model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1))

                precictions.append(prediction)
            action = np.argmax(np.array(precictions))
            game_action = self.get_game_action(snake, action - 1)
            done, _, snake, food = game.step(game_action)
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
        if self.network_type == 'tf':
            nn_model = self.model()
            nn_model.load(self.filename)

        elif self.network_type == 'torch':
            nn_model = torch.load('torch_model', map_location='cpu')

        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SnakeNN")

    parser.add_argument('--visualize', type=bool, default=True,
                        help='Visualize Snake Game')

    parser.add_argument('--network-type', type=str, choices=['torch', 'tf'],
                        default='torch', help='Framework Network type')

    parser.add_argument('--train', type=bool, default=True,
                        help='Framework Network type')

    parser.add_argument('--field-shape', type=tuple, default=(12, 12),
                        help='Field shape')

    args = parser.parse_args()
    network_type = args.network_type
    field_shape = args.field_shape
    visualize = args.visualize

    snake_nn = SnakeNN(field_shape=field_shape)

    if args.train:
        snake_nn.train()

    if visualize:
        snake_nn.visualise()