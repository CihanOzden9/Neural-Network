import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Ayarlar
BLOCK_SIZE = 20
SPEED = 50
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Pencere kapatma kontrolü
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Hareket et
        self._move(action) # AI komutunu uygula
        self.snake.insert(0, self.head)
        
        # 3. Ödül ve Çarpışma kontrolü
        reward = -0.01
        game_over = False
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10 # Çarpışma cezası
            return reward, game_over, self.score
            
        # 4. Yemek yeme
        if self.head == self.food:
            self.score += 1
            reward = 10 # Yemek ödülü
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. UI güncelle
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def _move(self, action):
        # action = [düz, sağa_dön, sola_dön]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # Değişiklik yok
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # Sağa dön
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Sola dön

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE

        self.head = Point(x, y)
        # BURADA RETURN OLMAMALI, SADECE KOORDİNAT GÜNCELLER

    def _is_collision(self):
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        if self.head in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()

    
if __name__ == '__main__':
    game = SnakeGameAI()
    while True:
        game_over, score = game.play_step()
        if game_over:
            break
    print('Oyun bitti, Skor:', score)