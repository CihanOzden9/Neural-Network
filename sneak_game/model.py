import numpy as np
import os

class ManualModel:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.lr = lr
        # Ağırlıkları rastgele başlat
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # x'i matris çarpımına uygun hale getir (1, 11)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self.input = x
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2 # Çıktı (Q değerleri)

    def train_step(self, state, target_q):
        # 1. Tahmin yap
        pred = self.forward(state)
        
        # 2. Hatayı hesapla (MSE Türevi)
        # target_q sadece eylemin olduğu indekste dolu gelir, diğerleri 0
        output_error = pred - target_q 
        
        # Sadece seçilen action için gradient uygula (basitleştirilmiş)
        # Backpropagation (Zincir kuralı)
        
        # Katman 2 için türevler
        dW2 = np.dot(self.a1.T, output_error)
        db2 = np.sum(output_error, axis=0, keepdims=True)
        
        # Katman 1 için türevler
        hidden_error = np.dot(output_error, self.W2.T)
        hidden_delta = hidden_error * (self.z1 > 0) # ReLU türevi (z > 0 ise 1, yoksa 0)
        
        dW1 = np.dot(self.input.T, hidden_delta)
        db1 = np.sum(hidden_delta, axis=0, keepdims=True)

        # 3. Ağırlıkları güncelle (Gradient Descent)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def save(self, file_name='model.npy'):
        if not os.path.exists('./model'):
            os.makedirs('./model')
        # Ağırlıkları tek dosyada sıkıştırıp kaydet
        np.savez(f'./model/{file_name}', W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, file_name='model.npy'):
        path = f'./model/{file_name}.npz'
        if os.path.exists(path):
            data = np.load(path)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            print("Hafıza yüklendi!")