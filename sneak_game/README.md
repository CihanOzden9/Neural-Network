# Autonomous Snake AI with Deep Analytics

Bu proje, Python ve Pygame kullanılarak geliştirilmiş, kendi kendine öğrenen otonom bir Yılan (Snake) oyunudur. Yapay zeka, bir Sinir Ağı (Neural Network) ve Pekiştirmeli Öğrenme (Reinforcement Learning) kullanarak yılanı kontrol eder. Proje, eğitim sürecini takip etmek ve analiz etmek için gelişmiş veri kaydı ve görselleştirme araçları içerir.

## Özellikler

*   **Otonom Oynanış**: Yılan, ` ManualModel` sinir ağı tarafından kontrol edilir ve zamanla oyunu öğrenir.
*   **Veri Odaklı Yapı**: Tüm eğitim verileri (model ağırlıkları, skorlar, pozisyonlar) `data/` klasöründe otomatik olarak saklanır.
*   **Kaldığı Yerden Devam Etme (Resume)**: `agent.py` içindeki `RESUME` bayrağı ile eğitimi durdurup, modelin ağırlıklarını koruyarak daha sonra devam edebilirsiniz.
*   **Canlı Grafik**: Eğitim sırasında anlık skor değişimlerini gösteren canlı bir grafik penceresi (`helper.py`).
*   **Detaylı Analiz Paneli**: Eğitim sonrası performans analizi için `dashboard.py` ile oluşturulan 4 panelli görselleştirme:
    *   Skor Geçmişi & Rekor Gelişimi
    *   Ölüm Nedenleri Analizi (Duvar vs. Kendine Çarpma)
    *   Verimlilik Analizi (Yemek başına atılan adım sayısı)
    *   Hareket Isı Haritası (Yılanın en çok gezdiği bölgeler)
*   **Optimize Edilmiş Performans**: 60 FPS sabit hızda akıcı eğitim simülasyonu.

## Kurulum

1.  Python kurulu olduğundan emin olun (Önerilen: 3.9+).
2.  Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Yapay Zekayı Eğitme

Eğitimi başlatmak için `agent.py` dosyasını çalıştırın:

```bash
python agent.py
```

*   **Yeni Eğitim**: Sıfırdan başlamak için `agent.py` dosyasını açın ve `RESUME = False` yapın. Bu işlem eski logları siler ve yeni bir model başlatır.
*   **Devam Etme**: Eğitime kaldığınız yerden devam etmek için `RESUME = True` yapın. Mevcut `data/model_weights.npz` dosyası yüklenir.

### 2. Analiz Panelini Görüntüleme

Eğitim verilerini analiz etmek için `dashboard.py` dosyasını çalıştırın:

```bash
python dashboard.py
```

Bu komut, `data/` klasöründeki CSV dosyalarını okuyarak performans grafiklerini içeren bir pencere açacaktır.

## Dosya Yapısı

*   `agent.py`: Ana yönetici dosya. Eğitimi döngüsünü, model yönetimini ve loglamayı kontrol eder.
*   `game.py`: Yılan oyunu mantığı ve grafik arayüzü (Pygame).
*   `model.py`: Sinir ağı mimarisi (NumPy tabanlı) ve ağırlık kaydetme/yükleme işlemleri.
*   `helper.py`: Eğitim sırasındaki canlı grafik çizim fonksiyonları.
*   `dashboard.py`: Eğitim sonrası detaylı veri analizi ve görselleştirme aracı.
*   `data/`: Tüm çıktıların (model, loglar) saklandığı klasör.
