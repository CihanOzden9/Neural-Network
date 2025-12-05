import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import ManualModel

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Rastgelelik oranı
        self.gamma = 0.9 # Gelecek ödülün önemi (Discount rate)
        self.memory = deque(maxlen=MAX_MEMORY)
        # 11 Giriş (Durum), 256 Gizli Nöron, 3 Çıkış (Hareket)
        self.model = ManualModel(11, 256, 3, LR)
        
        # Varsa eski hafızayı yükle (Hatırlama özelliği)
        try:
            self.model.load()
        except:
            pass

    def get_state(self, game):
        head = game.snake[0]
        # Yılanın etrafındaki noktalar
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # 1. Tehlike Düz mü?
            (dir_r and game._is_collision(point_r)) or 
            (dir_l and game._is_collision(point_l)) or 
            (dir_u and game._is_collision(point_u)) or 
            (dir_d and game._is_collision(point_d)),

            # 2. Tehlike Sağda mı?
            (dir_u and game._is_collision(point_r)) or 
            (dir_d and game._is_collision(point_l)) or 
            (dir_l and game._is_collision(point_u)) or 
            (dir_r and game._is_collision(point_d)),

            # 3. Tehlike Solda mı?
            (dir_d and game._is_collision(point_r)) or 
            (dir_u and game._is_collision(point_l)) or 
            (dir_r and game._is_collision(point_u)) or 
            (dir_l and game._is_collision(point_d)),
            
            # 4. Hareket Yönü
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # 5. Yemek Nerede?
            game.food.x < game.head.x, # Yemek solda
            game.food.x > game.head.x, # Yemek sağda
            game.food.y < game.head.y, # Yemek yukarıda
            game.food.y > game.head.y  # Yemek aşağıda
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
        # Q-Learning Algoritması (Bellman Denklemi)
        target = self.model.forward(state)
        target = target.copy() # Kopya oluştur (hata almamak için)
        
        Q_new = reward
        if not done:
            # Gelecekteki maksimum ödülü tahmin et ve ekle
            Q_new = reward + self.gamma * np.max(self.model.forward(next_state))

        target[0][np.argmax(action)] = Q_new
        self.model.train_step(state, target)

    def get_action(self, state):
        # Keşif vs Sömürü (Exploration vs Exploitation)
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        
        # Başlangıçta rastgele hareket et (öğrenmek için)
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Öğrendiğini uygula
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
    
    # game._is_collision metodunu agent için modifiye etmemiz gerekebilir
    # Şimdilik game.py içindeki metod Point kabul etmiyor olabilir, onu düzeltelim:
    # Basit bir yama fonksiyonu:
    original_collision = game._is_collision
    def collision_wrapper(pt=None):
        if pt is None: return original_collision()
        # Sanal çarpışma kontrolü
        if pt.x > game.w - 20 or pt.x < 0 or pt.y > game.h - 20 or pt.y < 0: return True
        if pt in game.snake[1:]: return True
        return False
    game._is_collision = collision_wrapper

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Kısa süreli hafıza (anlık öğrenme)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Oyun bitti, uzun süreli hafızayı eğit (Replay Memory)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Oyun: {agent.n_games}, Skor: {score}, Rekor: {record}')

if __name__ == '__main__':
    train()