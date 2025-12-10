import numpy as np
import os

class ManualModel:
    def __init__(self, input_size, hidden_sizes, output_size, lr=0.001):
        self.lr = lr
        self.layers = []
        self.biases = []
        
        # Layer dimensions: [input, hidden1, hidden2, ..., output]
        layer_dims = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_dims) - 1):
            # Weights: (in, out)
            W = np.random.randn(layer_dims[i], layer_dims[i+1]) * 0.1
            b = np.zeros((1, layer_dims[i+1]))
            self.layers.append(W)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self.activations = [x] # a0 (input), a1, a2...
        self.z_values = []     # z1, z2...
        
        curr_input = x
        
        # Iterate through all layers except the last (Hidden Layers)
        for i in range(len(self.layers) - 1):
            z = np.dot(curr_input, self.layers[i]) + self.biases[i]
            a = self.relu(z)
            self.z_values.append(z)
            self.activations.append(a)
            curr_input = a
            
        # Output Layer (Linear Activation)
        last_idx = len(self.layers) - 1
        z_out = np.dot(curr_input, self.layers[last_idx]) + self.biases[last_idx]
        self.z_values.append(z_out)
        self.activations.append(z_out) # Final output is also stored as activation for consistency
        
        return z_out

    def train_step(self, state, target_q):
        # 1. Forward Pass
        pred = self.forward(state)
        
        # 2. Backward Pass
        output_error = pred - target_q
        
        # Calculate Loss (MSE)
        # We only care about the error at the action index, but since target_q 
        # is constructed to match pred at other indices, this is effectively:
        # Loss = (Q_target - Q_pred)^2
        # However, target_q has values only at action index? 
        # No, in agent.py we do: target = pred.copy(); target[action] = Q_new
        # So error is 0 everywhere except the action index.
        loss = np.mean(np.square(output_error))

        # Store gradients to update later
        grads_W = [None] * len(self.layers)
        grads_b = [None] * len(self.biases)
        
        # Backpropagate error
        delta = output_error
        
        # Loop backwards
        for i in reversed(range(len(self.layers))):
            input_to_layer = self.activations[i]
            
            dW = np.dot(input_to_layer.T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            grads_W[i] = dW
            grads_b[i] = db
            
            if i > 0:
                error_prev = np.dot(delta, self.layers[i].T)
                relu_deriv = (self.z_values[i-1] > 0).astype(float)
                delta = error_prev * relu_deriv
                
        # 3. Update Weights
        for i in range(len(self.layers)):
            self.layers[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]
            
        return loss

    def save(self, file_name='model_weights.npz'):
        folder_path = './data'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        file_path = os.path.join(folder_path, file_name)
        
        # Pack everything into a dictionary
        save_dict = {}
        for i, (W, b) in enumerate(zip(self.layers, self.biases)):
            save_dict[f'W{i+1}'] = W
            save_dict[f'b{i+1}'] = b
            
        np.savez(file_path, **save_dict)
        # print("Model saved!") 

    def load(self, file_name='model_weights.npz'):
        path = f'./data/{file_name}'
        
        if os.path.exists(path):
            try:
                data = np.load(path)
                self.layers = []
                self.biases = []
                
                i = 1
                while f'W{i}' in data:
                    self.layers.append(data[f'W{i}'])
                    self.biases.append(data[f'b{i}'])
                    i += 1
                print(f"Model loaded successfully with {len(self.layers)} layers from {path}!")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"No existing model found at {path}, starting fresh.")
            return False