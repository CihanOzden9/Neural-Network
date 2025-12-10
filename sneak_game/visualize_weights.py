import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_w1():
    path = './model/model.npz'
    if not os.path.exists(path):
        print(f"Model file not found at {path}. Train the agent first!")
        return

    try:
        data = np.load(path)
        # Check for W1 specifically (old format) or W1 from dynamic format
        if 'W1' in data:
            w1 = data['W1']
        else:
            # If saved with dynamic keys W1, W2...
            # The keys might be 'arr_0' if not named explicitly in save, 
            # but our refactored save() uses explicit keys 'W1', 'b1' etc.
            keys = data.files
            if 'W1' in keys:
                w1 = data['W1']
            else:
                print(f"Could not find 'W1' in {keys}")
                return

        print(f"Loaded W1 with shape: {w1.shape}")
        
        # Plotting
        plt.figure(figsize=(10, 6))
        # Transpose so inputs are on Y-axis (easier to read if many inputs) or X-axis
        # Shape is (11, HiddenSize). Let's plot (HiddenSize, 11) or keep as is.
        # Usually: Rows=Features, Cols=Neurons.
        
        plt.imshow(w1, cmap='viridis', aspect='auto')
        plt.colorbar(label='Weight Value')
        plt.title('Visualization of Input Weights (W1)')
        plt.ylabel('Input Features (11)')
        plt.xlabel(f'Hidden Neurons ({w1.shape[1]})')
        
        # State feature labels based on agent.py
        feature_labels = [
            "Danger Straight", "Danger Right", "Danger Left",
            "Dir Left", "Dir Right", "Dir Up", "Dir Down",
            "Food Left", "Food Right", "Food Up", "Food Down"
        ]
        
        plt.yticks(ticks=np.arange(11), labels=feature_labels)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error loading validation: {e}")

if __name__ == "__main__":
    visualize_w1()
