
---

```markdown
# 🕊️ MobileNetV2 ile Drone ve Kuş Görüntü Sınıflandırması

Bu proje, transfer öğrenme yöntemiyle **drone** ve **kuş** görüntülerini sınıflandıran bir derin öğrenme modelidir. Temel olarak **MobileNetV2** mimarisi kullanılmaktadır.

---

## 📦 Veri Kümesi

- Kaynak: [Kaggle - Drone vs Bird Dataset](https://www.kaggle.com/datasets/muhammadsaoodsarwar/drone-vs-bird)
- Yapı:
  - `train/` → `bird/` ve `drone/` klasörlerine ayrılmış eğitim verileri
  - `val/` → %20 doğrulama verisi
  - `custom_test/` → elle toplanan gerçek dünya test görselleri

---

## 📌 Proje Özeti

- 🔍 **Amaç:** Bir görselin kuş mu yoksa drone mu olduğunu sınıflandırmak
- 🧠 **Model:** MobileNetV2 (ImageNet ön eğitimli, fine-tuning yapıldı)
- 🧪 **Doğruluk Oranları:**
  - Eğitim: **%99.88**
  - Doğrulama: **%99.88**
  - Gerçek dünya testi: **10 görüntüde 10 doğru tahmin**

---

## 📁 Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `drone_vs_bird_classifier.ipynb` | Modelin eğitim ve test kodlarını içeren notebook |
| `best_model.pt` | Eğitilmiş modelin ağırlık dosyası |
| `README.md` | İngilizce açıklama dosyası |
| `README_TR.md` | Bu dosya – Türkçe açıklama |

---

## 🔧 Kullanım

### Modeli yükle:
```python
import torch
from torchvision import models
import torch.nn as nn

model = models.mobilenet_v2(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 2)
)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

#Görsel Tahmin
from PIL import Image
from torchvision import transforms

img = Image.open("gorsel.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

print(predicted.item())  # 0: kuş, 1: drone



