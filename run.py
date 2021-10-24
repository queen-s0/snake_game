import argparse

from models.nn import SnakeNN
from models.value_iteration import ValueIterationSnake
from models.policy_iteration import PolicyIterationSnake

def get_args_parser():
    parser = argparse.ArgumentParser(description='Run Snake game agent', add_help=False)
    parser.add_argument('--algo', default=None, type=str,
                        choices = ['value iteration','policy iteration','mlp'],
                        help='name of the algorithm')
    parser.add_argument('--field-shape', default=(12, 12), nargs='+', type=int,
                        help='Size of the game field')
    parser.add_argument('--train-games', default=15000, type=int,
                        help='Number of initial games to create train dataset')
    parser.add_argument('--visualize', default=True, type=bool,
                        help='Visualize game')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--test-games', default=5, type=int,
                        help='Number of test games')

    return parser

def main(args):
    algorithm_name = args.algo
    field_shape = args.field_shape
    train_games = args.train_games
    test_games = args.test_games
    visualize = args.visualize
    seed = args.seed

    if algorithm_name == 'value iteration':
        snake = ValueIterationSnake(field_shape=field_shape,
                                    initial_games=train_games,
                                    test_games=test_games,
                                    visualize=visualize,
                                    seed=seed)
        snake.run()

    elif algorithm_name == 'policy iteration':
        snake = PolicyIterationSnake(field_shape=field_shape,
                                     initial_games=train_games,
                                     test_games=test_games,
                                     visualize=visualize,
                                     seed=seed)
        snake.run()

    elif algorithm_name == 'mlp':
        snake = SnakeNN(field_shape=field_shape,
                        initial_games=train_games,
                        test_games=test_games,
                        visualize=visualize,
                        seed=seed)
        snake.train()
        snake.visualise()

    else:
        raise NotImplementedError('Unknown type of algorithm')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Snake game agent', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)



