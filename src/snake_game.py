# -*- coding: utf-8 -*- 
import curses
import time
from random import randint
from src.render import Render

class SnakeGame:
    def __init__(self, board_width = 20, board_height = 20, gui = False, title=None):
        self.score = 0
        self.done = False
        self.board = {'width': board_width, 'height': board_height}
        self.gui = gui
        self.title = title

    def start(self):
        self.snake_init()
        self.generate_food()
        if self.gui: self.render_init()
        return self.generate_observations()

    def snake_init(self):
        x = randint(5, self.board["width"] - 5)
        y = randint(5, self.board["height"] - 5)
        self.snake = []
        vertical = randint(0,1) == 0
        for i in range(3):
            point = [x + i, y] if vertical else [x, y + i]
            self.snake.insert(0, point)

    def generate_food(self):
        food = []
        while food == []:
            food = [randint(0, self.board["width"] - 1), randint(0, self.board["height"] - 1)]
            if food in self.snake: food = []
        self.food = food

    def render_init(self):
        self.render_obj = Render(shape=(self.board['width'], self.board['height']), title=self.title)
        self.render()

    def render(self):
        self.render_obj.update(self.snake, self.food, self.score)
        time.sleep(0.2)

    def step(self, key):
        # 0 - UP
        # 1 - RIGHT
        # 2 - DOWN
        # 3 - LEFT
        if self.done == True:
            self.end_game()
        self.create_new_point(key)
        if self.food_eaten():
            self.score += 1
            self.generate_food()
        else:
            self.remove_last_point()
        self.check_collisions()
        if self.gui: self.render()
        return self.generate_observations()

    def create_new_point(self, key):
        new_point = [self.snake[0][0], self.snake[0][1]]
        if key == 0:
            new_point[0] -= 1
        elif key == 1:
            new_point[1] += 1
        elif key == 2:
            new_point[0] += 1
        elif key == 3:
            new_point[1] -= 1
        self.snake.insert(0, new_point)

    def remove_last_point(self):
        self.snake.pop()

    def food_eaten(self):
        return self.snake[0] == self.food

    def check_collisions(self):
        if (self.snake[0][0] == -1 or
            self.snake[0][0] == self.board["width"] or
            self.snake[0][1] == -1 or
            self.snake[0][1] == self.board["height"] or
            self.snake[0] in self.snake[1:-1]):
            self.done = True

    def generate_observations(self):
        return self.done, self.score, self.snake, self.food

    def render_destroy(self):
        self.render_obj.quit()

    def end_game(self):
        if self.gui: self.render_destroy()
        raise Exception("Game over")

if __name__ == "__main__":
    shape = (10, 10)
    game = SnakeGame(*shape, gui = True)
    game.start()
    for _ in range(25):
        game.step(randint(0,3))
