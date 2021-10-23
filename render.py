import pygame

bg_color = (0, 0, 0)
snake_color = (0, 0, 255)
head_color = (128, 128, 128)
food_color = (255, 0, 0)
text_color = (255, 255, 255)

class Render:
    def __init__(self, shape=(1000, 1000)):
        pygame.init()
        self.dis = pygame.display.set_mode(shape)
        pygame.display.set_caption('RL Snake Game')
        self.dis.fill(bg_color)
        pygame.display.update()
        self.score_font = pygame.font.SysFont("comicsansms", 35)
        
    def draw_snake(self, snake): #snake - list
        p = snake[0]
        pygame.draw.rect(self.dis, head_color, [p[1] * 50, p[0] * 50, 50, 50])
        for p in snake[1:]:
            pygame.draw.rect(self.dis, snake_color, [p[1] * 50, p[0] * 50, 50, 50])
        pygame.display.update()
        
    def draw_food(self, food):
        pygame.draw.circle(self.dis, food_color, (food[1] * 50, food[0] * 50), 25)
        pygame.display.update()
        
    def draw_score(self, score):
        value = self.score_font.render("Score: " + str(score), True, text_color)
        self.dis.blit(value, [0, 0])
        pygame.display.update()
        
    def update(self, snake, food, score):
        self.dis.fill(bg_color)
        self.draw_snake(snake)
        self.draw_food(food)
        self.draw_score(score)
        
    def quit(self):
        pygame.quit()