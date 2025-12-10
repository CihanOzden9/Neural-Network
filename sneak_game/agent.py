import os
import csv
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import ManualModel
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Training Control
RESUME = True

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 
        self.gamma = 0.9 
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = ManualModel(11, [256], 3, LR)
        
        # Ensure data directory exists
        if not os.path.exists('./data'):
            os.makedirs('./data')

        if RESUME:
            if self.model.load('model_weights.npz'):
                print("Resumed from data/model_weights.npz")
                 # Optional: Logic to determine n_games from logs could be added here
                 # but for now we just load weights.
            else:
                print("Could not load model, starting fresh.")
        else:
             # If not resuming, maybe clear old logs? 
             # User said: "initialize a fresh model... and start new log files (overwrite old ones)"
             if os.path.exists('./data/training_log.csv'):
                 os.remove('./data/training_log.csv')
             if os.path.exists('./data/positions_log.csv'):
                 os.remove('./data/positions_log.csv')
             print("Starting fresh. Old logs removed.")

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger Left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food
            game.food.x < game.head.x, 
            game.food.x > game.head.x, 
            game.food.y < game.head.y, 
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, done in mini_sample:
            self.train_short_memory(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = self.model.forward(state)
        target = target.copy()
        Q_new = reward
        if not done:
            Q_new = reward + self.gamma * np.max(self.model.forward(next_state))
        target[0][np.argmax(action)] = Q_new
        self.model.train_step(state, target)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            prediction = self.model.forward(state)
            move = np.argmax(prediction)
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    # Init Logs
    log_dir = './data'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_log_path = os.path.join(log_dir, 'training_log.csv')
    pos_log_path = os.path.join(log_dir, 'positions_log.csv')

    # If initializing fresh logs (RESUME=False handled in __init__ remove), assume need headers
    if not os.path.exists(train_log_path):
        with open(train_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Game_No', 'Score', 'Record', 'Avg_Steps', 'Death_Reason'])
    
    if not os.path.exists(pos_log_path):
        with open(pos_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Game_No', 'X', 'Y'])

    # Accumulators for steps in current game
    current_game_steps = 0

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        
        # Move
        reward, done, score, steps_per_food, death_reason = game.play_step(final_move, agent.n_games)
        state_new = agent.get_state(game)

        # Train Check
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        # Log Position
        with open(pos_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([agent.n_games, game.head.x, game.head.y])
            
        current_game_steps += 1

        if done:
            # Train Long
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}, Reason: {death_reason}')
            
            # Log Training
            # Avg_Steps: Total steps in game / (Score + 1) roughly or just Total Steps? User asked "Avg_Steps".
            # Usually means "Steps per Point" or "Total Steps". Let's log Total Steps / (Score+1) as efficiency metric.
            avg_steps = current_game_steps / (score + 1)
            
            with open(train_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([agent.n_games, score, record, f'{avg_steps:.2f}', death_reason])

            # Reset
            current_game_steps = 0
            
            # Plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()