# Snake Game 🐍

This final "Reinforcement Learning course" project focuses on the artificial intelligence of the **"Snake Game"**. Essence of the game is a piece of cake: snake eats the food that arises on the field and fill the map of its bodies as soon as possible. 

We have implemented three algorithms:
* Value Iteration
* Policy Iteration
* Simple Multi-Layer perceptron

In order to estimate the perfomance of each algorithm we have conducted an experiments, using the following metrics:
* **Average number of steps (ANS)**: How many steps does the Snake take on average before the collision;
* **Average maximal Snake length (AML)**: Average maximal length of the snake during the game;

Our video-presentation can be found [here](https://youtu.be/3HkdobjWHPA).

## Results

**Algorithm** |       **Value Iteration**             |  **Policy Iteration**   |   **MLP**
:------------------------:|:-------------------------:|:-------------------------:|:--------------------------:|
**Demo (Non-optimal)** | ![](gifs/value-iteration.gif)  |  ![](gifs/policy-iteration.gif) | ![](gifs/mlp.gif) 
**Average number of steps** |  **424** |  **198** | **123**
**Average maximal length** | **22** | **23** | **12**

## Repository structure

- ```models/``` — contains algorithms \
    ```├── nn.py``` — MLP approach\
    ```├── policy_iteration.py``` — Policy Iteration algorithm\
    ```└── value_iteration.py``` — Value Iteration
- ```src/``` — auxiliary instrumets \
    ```├── render.py``` — Snake Game rendering\
    ```└── snake_game.py``` — logic of the Snake Game
- ```gifs/``` — contains gifs 
- ```supplementary/``` — contains presentation in `.pdf` format
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
