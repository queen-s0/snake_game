import pygame

BG_COLOR = (255, 255, 255)
SNAKE_COLOR = (0, 102, 0)
HEAD_COLOR = (0, 0, 0)
FOOD_COLOR = (255, 0, 0)
TEXT_COLOR = (0, 0, 0)
SCALE = 40
COUNTER_SIZE = 30

class Render:
    def __init__(self, shape=(20, 20), title='RL Game'):

        self.title = title
        pygame.init()
        self.initial_width, self.initial_height = shape
        self.coef_width, self.coef_height = [x * SCALE for x in shape]

        self.dis = pygame.display.set_mode((self.coef_width, self.coef_height))
        pygame.display.set_caption(f'RL Snake Game. {title} algorithm.')
        self.dis.fill(BG_COLOR)
        pygame.display.update()
        self.score_font = pygame.font.SysFont("comicsansms", COUNTER_SIZE)
        self.step = 0

    def draw_snake(self, snake): #snake - list
        p = snake[0]
        pygame.draw.rect(self.dis, HEAD_COLOR, [p[1] * SCALE, p[0] * SCALE, SCALE, SCALE])
        print('Step: ', self.step, 'Head coord: ', p)
        for p in snake[1:]:
            print('Step: ', self.step, 'Snake coord: ', p)

            x = p[1] * SCALE
            y = p[0] * SCALE
            pygame.draw.rect(self.dis, SNAKE_COLOR, [x, y, SCALE, SCALE])

            rect = pygame.Rect(x, y, SCALE, SCALE)
            pygame.draw.rect(self.dis, (0, 0, 0), rect, 1)

    def draw_food(self, food):
        print('Food coord:', food)
        pygame.draw.circle(self.dis, FOOD_COLOR, (food[1] * SCALE + SCALE / 2, food[0] * SCALE + SCALE / 2), SCALE / 2)

    def draw_score(self, score):
        value = self.score_font.render("Score: " + str(score) + f'. Step: {self.step}', True, TEXT_COLOR)
        self.dis.blit(value, [0, 0])
        print("Score: " + str(score) + f'. Step: {self.step}')

    def draw_grid(self):
        for x in range(0, self.coef_width, SCALE):
            for y in range(0, self.coef_height, SCALE):
                rect = pygame.Rect(x, y, SCALE, SCALE)
                pygame.draw.rect(self.dis, (255, 153, 204), rect, 1)
        
    def update(self, snake, food, score):
        self.dis.fill(BG_COLOR)
        self.draw_grid()
        self.draw_snake(snake)
        self.draw_food(food)
        self.draw_score(score)
        pygame.display.update()
        self.step += 1
        
    def quit(self):
        pygame.quit()