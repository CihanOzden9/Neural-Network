import numpy as np

# Dosya yolunu tam olarak yaz
path = './model/model.npy.npz' 

try:
    data = np.load(path)
    
    print("Dosya içindeki anahtarlar:", data.files)
    
    print("\n--- W1 (Giriş Katmanı Ağırlıkları) ---")
    print(data['W1']) # İlk katman ağırlıkları
    
    print("\n--- W1 Boyutu ---")
    print(data['W1'].shape) # Örn: (11, 256)

except FileNotFoundError:
    print(f"Dosya bulunamadı: {path}")