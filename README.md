# Snake Game

This final "Reinforcement Learning course" project focuses on the artificial intelligence of the **"Snake Game"**. Essence of the game is a piece of cake: snake eats the food that arises on the field and fill the map of its bodies as soon as possible. 

We have implemented three algorithms:
* Value Iteration
* Policy Iteration
* Simple Multi-Layer perceptron

In order to estimate the perfomance of each algorithm we have conducted an experiments, using the following metrics:
* **Steps per food**: How many steps it took on average for the snake to eat food;
* **Average length**: Average length of the snake during the game;

## Repository structure

- ```models/``` — contains algorithms \
    ```├── nn.py``` — MLP approach\
    ```├── policy_iteration.py``` — Policy Iteration algorithm\
    ```└── value_iteration.py``` — Value Iteration
- ```src/``` — auxiliary instrumets \
    ```├── render.py``` — Snake Game rendering\
    ```└── snake_game.py``` — logic of the Snake Game
- ```run.py``` — start experiments

## Installation

Follow the rows:

```
$ git clone https://github.com/queen-s0/snake_game.git
$ cd snake_game
$ conda env create -f environment.yml
```

## Usage
```
usage: run.py [-h] [--algo {value iteration,policy iteration,mlp}]
              [--field-shape FIELD_SHAPE [FIELD_SHAPE ...]]
              [--train-games TRAIN_GAMES] [--visualize VISUALIZE]
              [--seed SEED] [--test-games TEST_GAMES]
```

## Example
```
python run.py --algo 'mlp' --field-shape 15 15 --train-games 8000 --visualize True --test-games 5
```

## Results

### MLP

[comment]: <> (![MLP]&#40;gifs/mlp.gif&#41;)

[comment]: <> ([<img src="gifs/mlp.gif" width="250"/>]&#40;gifs/mlp.gif&#41;)

<p float="middle">
  <img src="gifs/value-iteration.gif" width="250" />
  <img src="gifs/policy-iteration.gif" width="250" /> 
  <img src="gifs/mlp.gif" width="250" />
</p>