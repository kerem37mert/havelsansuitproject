## Teknik Rapor

---

## İçindekiler

1. [Projeye Genel Bakış](#1-projeye-genel-bakış)
2. [Veri Setleri](#2-veri-setleri)
   - 2.1 MRL Eye Dataset
   - 2.2 YawDD Dataset
3. [Model 1: Göz Durumu Tespiti (mrl_eye1.ipynb)](#3-model-1-göz-durumu-tespiti)
   - 3.1 Ortam Hazırlığı
   - 3.2 Veri Yükleme ve Doğrulama
   - 3.3 Veri Artırma (Data Augmentation)
   - 3.4 Model Mimarisi
   - 3.5 Derleme ve Eğitim Yapılandırması
   - 3.6 Eğitim Süreci
   - 3.7 Değerlendirme
   - 3.8 Model Kaydı
4. [Model 2: Esnerme Tespiti (yawdd.ipynb)](#4-model-2-esnerme-tespiti)
   - 4.1 Ortam Hazırlığı
   - 4.2 Veri Seti Keşfi
   - 4.3 Ağız Açıklık Oranı (MAR) Hesaplama
   - 4.4 Eşik Belirleme (Otomatik Etiketleme)
   - 4.5 Frame Çıkarma ve Veri Seti Oluşturma
   - 4.6 Veri Artırma
   - 4.7 Model Mimarisi
   - 4.8 Derleme ve Eğitim Yapılandırması
   - 4.9 Eğitim Süreci
   - 4.10 Değerlendirme
   - 4.11 Eşik Değerleri ve Model Kaydı
5. [Nihai Sistem Mimarisi ve Dosya Yapısı](#5-nihai-sistem-mimarisi)
6. [config.py — Sistem Sabitleri](#6-configpy)
7. [models.py — Model Yükleme ve Çıkarım](#7-modelspy)
8. [detector.py — Yüz Tespiti ve Özellik Çıkarımı](#8-detectorpy)
9. [fusion.py — Yorgunluk Skoru Hesaplama](#9-fusionpy)
10. [main.py — Gerçek Zamanlı Döngü ve Görselleştirme](#10-mainpy)
11. [Özellikler ve Çıkarım Pipeline'ı](#11-özellikler-ve-çıkarım-pipelineı)
12. [Yorgunluk Skoru Matematiksel Modeli](#12-yorgunluk-skoru-matematiksel-modeli)
13. [Kullanılan Teknolojiler](#13-kullanılan-teknolojiler)

---

## 1. Projeye Genel Bakış

Bu proje, araç sürücülerinin gerçek zamanlı olarak yorgunluk/uyuklama durumunun tespitine yönelik çok modlu (multi-modal) bir derin öğrenme sistemidir. Sistem iki ayrı eğitilmiş CNN modeli ile MediaPipe'ın yüz landmark tespitini bir araya getirerek anlık webcam akışından üç farklı biyometrik sinyali eş zamanlı olarak analiz eder:

1. **PERCLOS (Percentage Eye Closure):** Göz kapanma oranı — yorgunluk literatüründe en kanıtlanmış gösterge
2. **Esnerme (Yawning):** Ağız açılma davranışı — güçlü bir tamamlayıcı sinyal
3. **Baş Pozisyonu (Head Pose):** Öne eğilme açısı (pitch) — ileri yorgunluğun fiziksel belirtisi

Bu üç sinyal ağırlıklı bir skor sistemiyle birleştirilir ve Üstel Hareketli Ortalama (EMA) ile yumuşatılarak 0–100 arası bir "yorgunluk skoru" elde edilir. Skor üç seviyeye ayrılır: **UYANIK** (0–30), **UYARI** (30–60), **UYUKLUYOR** (60–100).

---

## 2. Veri Setleri

### 2.1 MRL Eye Dataset

**Tam Adı:** Machine Learning and Reasoning Lab (MRL) Eye Dataset
**Kaynak:** Ostrava Teknik Üniversitesi, Çek Cumhuriyeti
**İndirme:** `http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip` / Kaggle: `imadeddinedjerarda/mrl-eye-dataset`

#### Genel İstatistikler

| Bölüm | Uyanık | Uykulu | Toplam |
|-------|--------|--------|--------|
| Eğitim (Train) | 25.770 | 25.167 | 50.937 |
| Doğrulama (Val) | 8.591 | 8.389 | 16.980 |
| Test | 8.591 | 8.390 | 16.981 |
| **Toplam** | **42.952** | **41.946** | **84.898** |

Sınıf dağılımı oldukça dengeli olup herhangi bir sınıf ağırlıklandırması (class weighting) gerekmemiştir.

#### Veri Seti Özellikleri

Her görüntü kızılötesi (infrared) sensörle çekilmiş tek bir göz bölgesini içerir. Dosya adlarında şifrelenmiş meta veriler bulunmaktadır:

| Alan | Değerler |
|------|---------|
| Özne ID (subject_id) | 37 benzersiz denek |
| Göz durumu (eye_state) | 0 = Uykulu, 1 = Uyanık |
| Cinsiyet | 0 = Erkek, 1 = Kadın |
| Gözlük | 0 = Yok, 1 = Var |
| Yansıma | 0 = Yok, 1 = Az, 2 = Fazla |
| Aydınlatma | 0 = Kötü, 1 = İyi |
| Sensör ID | 01 = Intel RealSense 640×480, 02 = IDS 1280×1024, 03 = Aptina 752×480 |

Bu çeşitlilik (farklı sensörler, aydınlatma koşulları, demografik gruplar) modelin genellenebilirliğini önemli ölçüde artırmaktadır.

#### Dizin Yapısı

```
data/mrl_eye/
├── train/
│   ├── awake/    (25.770 görüntü)
│   └── sleepy/   (25.167 görüntü)
├── val/
│   ├── awake/    (8.591 görüntü)
│   └── sleepy/   (8.389 görüntü)
└── test/
    ├── awake/    (8.591 görüntü)
    └── sleepy/   (8.390 görüntü)
```

---

### 2.2 YawDD Dataset

**Tam Adı:** Yawning Detection Dataset
**Kaynak:** Otoyol sürücü izleme sistemi simülasyonu
**Format:** Video dosyaları (Dash kamera kayıtları)

#### Veri Seti İçeriği

| Kategori | Video Sayısı |
|----------|-------------|
| Kadın sürücü | 13 video |
| Erkek sürücü | 16 video |
| **Toplam** | **29 video** |

Videolar araç içi kameradan alınan gerçekçi sürüş koşullarını içermektedir. Her video birden fazla davranış türünü (normal sürüş, esnerme, konuşma vb.) barındırmaktadır.

Ham veriden otomatik etiketleme yapılmıştır — yani insan etiketlemesi yoktur, Ağız Açıklık Oranı (MAR) hesabına dayalı istatistiksel bir etiketleme yöntemi kullanılmıştır (detaylar Bölüm 4'te).

---

## 3. Model 1: Göz Durumu Tespiti

**Notebook:** `mrl_eye1.ipynb`
**Çalışma Ortamı:** Google Colab (GPU etkin, Tesla T4/V100)
**Çıktı:** `models/mrl_eye_final_model.h5` (~11 MB)

### 3.1 Ortam Hazırlığı

```python
from google.colab import drive
drive.mount('/content/drive')
```

Google Drive bağlandıktan sonra Kaggle API aracılığıyla MRL Eye Dataset indirilmiştir:

```bash
pip install -q kaggle
# Kaggle API anahtarı yapılandırması
kaggle datasets download -d imadeddinedjerarda/mrl-eye-dataset
unzip mrl-eye-dataset.zip -d /content/drive/MyDrive/havelsandataset/mrl-eye-dataset/
```

Veri seti hız için yerel diske kopyalanmıştır (Drive I/O gecikmesini azaltmak amacıyla):

```python
import shutil
shutil.copytree(
    '/content/drive/MyDrive/havelsandataset/mrl-eye-dataset/',
    '/content/mrl_eye/'
)
```

Ortam kontrolü:
```
TensorFlow version: 2.19.0
GPU available: True
```

### 3.2 Veri Yükleme ve Doğrulama

```python
import os

train_awake  = len(os.listdir('/content/mrl_eye/train/awake'))   # 25.770
train_sleepy = len(os.listdir('/content/mrl_eye/train/sleepy'))  # 25.167
val_awake    = len(os.listdir('/content/mrl_eye/val/awake'))     # 8.591
val_sleepy   = len(os.listdir('/content/mrl_eye/val/sleepy'))    # 8.389
test_awake   = len(os.listdir('/content/mrl_eye/test/awake'))    # 8.591
test_sleepy  = len(os.listdir('/content/mrl_eye/test/sleepy'))   # 8.390
```

### 3.3 Veri Artırma (Data Augmentation)

Model genellenebilirliğini artırmak için eğitim sırasında gerçek zamanlı veri artırma uygulanmıştır:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,           # [0,255] → [0,1] normalizasyon
    rotation_range=15,        # ±15 derece döndürme
    width_shift_range=0.10,   # Yatay öteleme ±%10
    height_shift_range=0.10,  # Dikey öteleme ±%10
    horizontal_flip=True,     # Yatay çevirme
    zoom_range=0.10           # ±%10 yakınlaştırma/uzaklaştırma
)

val_test_datagen = ImageDataGenerator(rescale=1./255)  # Sadece normalizasyon
```

**Neden bu augmentasyonlar?**
- *Rotation ±15°:* Gerçek kullanımda kameraya göre baş hafifçe dönemez, ama göz görüntüsü MediaPipe crop'u sırasında küçük açısal farklılıklar içerebilir
- *Shifts ±10%:* Landmark tespitindeki küçük hataların bounding box'a yansımasını simüle eder
- *Horizontal flip:* Sol/sağ göz simetrisini kullanarak veri setini çeşitlendirir
- *Zoom ±10%:* Farklı mesafelerden çekilmiş görüntülere benzer varyasyon

Doğrulama ve test setlerinde yalnızca normalizasyon uygulanmış, augmentation yapılmamıştır.

```python
IMG_HEIGHT = 84
IMG_WIDTH  = 84
BATCH_SIZE = 128

train_generator = train_datagen.flow_from_directory(
    '/content/mrl_eye/train',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='binary'
)
```

Görüntüler 84×84 piksel olarak yeniden boyutlandırılmıştır. MRL veri seti kızılötesi görüntüler içermesine rağmen RGB olarak yüklenmiştir; bu durum modelin çıkarım aşamasında (webcam BGR→RGB dönüşümü) tutarlılığı sağlar.

### 3.4 Model Mimarisi

Dört evrişim bloğu ve iki tam bağlantılı katmandan oluşan bir CNN kullanılmıştır:

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    # ── Blok 1 ──────────────────────────────────────
    layers.Conv2D(64, (3, 3), activation='relu',
                  input_shape=(84, 84, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # ── Blok 2 ──────────────────────────────────────
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # ── Blok 3 ──────────────────────────────────────
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # ── Blok 4 ──────────────────────────────────────
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # ── Sınıflandırıcı ──────────────────────────────
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')   # İkili çıktı
])
```

#### Mimari Detayları

| Katman | Filtre/Nöron | Aktivasyon | Çıktı Boyutu | Parametre |
|--------|-------------|------------|--------------|-----------|
| Conv2D | 64 × (3×3) | ReLU | 82×82×64 | 1.792 |
| MaxPool | 2×2 | — | 41×41×64 | 0 |
| Dropout | 0.25 | — | 41×41×64 | 0 |
| Conv2D | 128 × (3×3) | ReLU | 39×39×128 | 73.856 |
| MaxPool | 2×2 | — | 19×19×128 | 0 |
| Dropout | 0.25 | — | 19×19×128 | 0 |
| Conv2D | 256 × (3×3) | ReLU | 17×17×256 | 295.168 |
| MaxPool | 2×2 | — | 8×8×256 | 0 |
| Dropout | 0.25 | — | 8×8×256 | 0 |
| Conv2D | 512 × (3×3) | ReLU | 6×6×512 | 1.180.160 |
| MaxPool | 2×2 | — | 3×3×512 | 0 |
| Dropout | 0.25 | — | 3×3×512 | 0 |
| Flatten | — | — | 4.608 | 0 |
| Dense | 256 | ReLU | 256 | 1.179.904 |
| Dropout | 0.50 | — | 256 | 0 |
| Dense | 128 | ReLU | 128 | 32.896 |
| Dropout | 0.30 | — | 128 | 0 |
| Dense | 1 | Sigmoid | 1 | 129 |
| **Toplam** | | | | **~2.764.000** |

**Mimari Tasarım Kararları:**

- Filtre sayısının bloktan bloğa ikiye katlanması (64→128→256→512): Düşük seviyeli özelliklerden (kenar, doku) yüksek seviyeli özelliklere (göz kapağı şekli, açıklık durumu) hiyerarşik öğrenme sağlar
- Her evrişim bloğuna Dropout(0.25): Overfitting'e karşı koruma — göz görüntüsü oldukça düşük çözünürlüklü ve gürültülüdür
- Dense katmanları arasında güçlü Dropout(0.5): Sınıflandırıcı başında aşırı öğrenmeyi engeller
- Son katmanda Sigmoid: İkili sınıflandırma (kapalı/açık) için olasılık çıkışı
- Batch Normalization kullanılmamıştır — MRL veri seti halihazırda normalize edilmiş kızılötesi görüntüler içerdiğinden gerekmemiştir

### 3.5 Derleme ve Eğitim Yapılandırması

```python
from tensorflow.keras import mixed_precision

# GPU bellek kullanımını optimize etmek için karışık hassasiyet
mixed_precision.set_global_policy('mixed_float16')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

**Mixed Precision (float16):** İleri/geri yayılım hesaplamaları float16 ile yapılır; ağırlıklar float32 olarak tutulur. Bu yaklaşım:
- GPU bellek kullanımını yaklaşık %40 azaltır
- Tensor Core'lar üzerinde 2–3× hızlanma sağlar
- Sayısal kararlılık için kritik işlemlerde float32'ye geri döner

#### Callback'ler

```python
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

callbacks = [
    ModelCheckpoint(
        '/content/drive/MyDrive/.../best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,           # 5 epoch iyileşme yoksa dur
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,           # LR'yi yarıya indir
        patience=3,           # 3 epoch iyileşme yoksa
        min_lr=1e-7
    )
]
```

### 3.6 Eğitim Süreci

```python
EPOCHS = 100

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)
```

- Maksimum 100 epoch planlanmış, EarlyStopping nedeniyle erken sonlanabilir
- Her epoch'ta tüm 50.937 eğitim görüntüsü işlenmiştir (batch_size=128 → ~398 adım/epoch)
- En iyi doğrulama doğruluğu ModelCheckpoint ile kaydedilmiştir

Eğitim sonunda accuracy ve loss eğrileri çizdirilmiştir:

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'],     label='Eğitim')
ax1.plot(history.history['val_accuracy'], label='Doğrulama')
ax1.set_title('Model Doğruluğu')
ax2.plot(history.history['loss'],     label='Eğitim')
ax2.plot(history.history['val_loss'], label='Doğrulama')
ax2.set_title('Model Kaybı')
```

### 3.7 Değerlendirme

Test seti değerlendirmesi:

```python
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
```

Detaylı metrikler için:

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_pred = (model.predict(test_generator) > 0.5).astype(int)
y_true = test_generator.classes

print(classification_report(y_true, y_pred,
      target_names=['Sleepy', 'Awake']))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Sleepy','Awake'],
            yticklabels=['Sleepy','Awake'])
```

Çıktı görselleştirmesi:

```python
# 16 test görüntüsü için tahmin gösterimi
# Yeşil çerçeve: Doğru tahmin
# Kırmızı çerçeve: Yanlış tahmin
# Üzerinde güven yüzdesi
```

### 3.8 Model Kaydı

```python
# Yerel kayıt
model.save('/content/mrl_eye_final_model.h5')

# Google Drive'a yedekleme
shutil.copy(
    '/content/mrl_eye_final_model.h5',
    '/content/drive/MyDrive/havelsandataset/mrl-eye-dataset/mrl_eye_final_model.h5'
)

# Yeniden yükleme kontrolü
model = tf.keras.models.load_model(
    '/content/drive/MyDrive/havelsandataset/mrl-eye-dataset/mrl_eye_final_model.h5'
)
```

Kaydedilen model: **`models/mrl_eye_final_model.h5`** (≈11 MB)

---

## 4. Model 2: Esnerme Tespiti

**Notebook:** `yawdd.ipynb`
**Çalışma Ortamı:** Google Colab (GPU etkin)
**Çıktı:** `models/yawning_model.h5` (~1.7 MB) + `models/yawning_thresholds.json`

### 4.1 Ortam Hazırlığı

```python
from google.colab import drive
drive.mount('/content/drive')

# Gerekli kütüphaneler
!pip install -q mediapipe
!pip install -q opencv-python-headless
!pip install -q tqdm

# MediaPipe FaceLandmarker modeli indir
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
    'face_landmarker/float16/1/face_landmarker.task',
    'face_landmarker.task'
)
```

### 4.2 Veri Seti Keşfi

```python
import os

DATASET_PATH = '/content/drive/MyDrive/havelsandataset/yawdd-dataset/Dash/Dash'

video_files = []
for gender in ['Female', 'Male']:
    gender_path = os.path.join(DATASET_PATH, gender)
    for f in os.listdir(gender_path):
        if f.endswith(('.avi', '.mp4')):
            video_files.append(os.path.join(gender_path, f))

# Female: 13 video, Male: 16 video → Toplam 29 video
print(f"Toplam video sayısı: {len(video_files)}")
```

### 4.3 Ağız Açıklık Oranı (MAR) Hesaplama

Esnermeyi otomatik olarak etiketlemek için Ağız Açıklık Oranı (Mouth Aspect Ratio — MAR) hesaplanmıştır. MAR, gözün açıklık oranını ölçen EAR (Eye Aspect Ratio) metriğinin ağıza uyarlanmış versiyonudur.

```python
# MediaPipe 478 noktalı yüz modelinde ağız landmarkları
MOUTH_LEFT   = 61    # Sol ağız köşesi
MOUTH_RIGHT  = 291   # Sağ ağız köşesi
MOUTH_TOP    = 13    # Üst dudak merkezi
MOUTH_BOTTOM = 14    # Alt dudak merkezi

def compute_mar(landmarks, img_w, img_h):
    """
    Ağız Açıklık Oranı hesaplar.

    MAR = dikey_mesafe / yatay_mesafe
        = |nokta_13 - nokta_14| / |nokta_61 - nokta_291|

    Normal ağız: MAR ≈ 0.05 – 0.15
    Esnerme:     MAR ≈ 0.30 ve üzeri
    """
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * img_w, lm.y * img_h])

    top    = pt(MOUTH_TOP)
    bottom = pt(MOUTH_BOTTOM)
    left   = pt(MOUTH_LEFT)
    right  = pt(MOUTH_RIGHT)

    vertical   = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    return vertical / (horizontal + 1e-6)
```

Ağız kırpma için ise 20 noktalı dış hat kullanılmıştır:

```python
MOUTH_OUTLINE = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,  # Üst dudak
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146  # Alt dudak
]
```

### 4.4 Eşik Belirleme (Otomatik Etiketleme)

Ham MAR dağılımından istatistiksel eşikler belirlenmiştir. Bu yaklaşım manuel etiketleme ihtiyacını ortadan kaldırır.

```python
# Tüm videolardan MAR değerleri toplanır
mar_values_all = []

for video_path in video_files[:5]:  # Eşik için ilk 5 video
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        result = face_landmarker.detect(mp_image)
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            h, w = frame.shape[:2]
            mar = compute_mar(lm, w, h)
            mar_values_all.append(mar)
    cap.release()

# İstatistiksel eşik belirleme
MAR_THRESHOLD  = np.percentile(mar_values_all, 82)   # → 0.3233
MAR_NORMAL_MAX = np.percentile(mar_values_all, 40)   # → 0.1459

# Etiketleme kuralı:
# MAR >= 0.3233          → "yawning"  (en yüksek %18)
# MAR <= 0.1459          → "normal"   (en düşük %40)
# 0.1459 < MAR < 0.3233 → DIŞLANDI   (belirsiz orta bölge)
```

**Eşiklerin Kaydedilmesi:**

```json
{
  "mar_yawning_threshold": 0.3232955900442968,
  "mar_normal_max":        0.1459407732341086,
  "img_size":              64,
  "class_indices": {
    "normal":  0,
    "yawning": 1
  }
}
```

Bu değerler `models/yawning_thresholds.json` dosyasına kaydedilmiştir.

### 4.5 Frame Çıkarma ve Veri Seti Oluşturma

```python
IMG_SIZE = 64           # Kırpılan ağız görüntüsü boyutu
FRAME_SKIP = 3          # Her 3 frame'den birini işle (temporal çeşitlilik)
MAX_PER_CLASS = 150     # Video başına her sınıftan maksimum görüntü sayısı

os.makedirs('/content/yawning_dataset/yawning', exist_ok=True)
os.makedirs('/content/yawning_dataset/normal',  exist_ok=True)

for video_path in video_files:
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    yawn_count = 0
    norm_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:  # Frame atlama
            continue

        result = face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            continue

        lm = result.face_landmarks[0]
        h, w = frame.shape[:2]
        mar = compute_mar(lm, w, h)

        # Etiketleme
        if mar >= MAR_THRESHOLD and yawn_count < MAX_PER_CLASS:
            label = 'yawning'
            yawn_count += 1
        elif mar <= MAR_NORMAL_MAX and norm_count < MAX_PER_CLASS:
            label = 'normal'
            norm_count += 1
        else:
            continue  # Belirsiz bölge veya limit doldu

        # Ağız bölgesini kırp ve kaydet
        mouth_crop = crop_mouth_region(frame, lm, w, h)
        mouth_resized = cv2.resize(mouth_crop, (IMG_SIZE, IMG_SIZE))

        filename = f"{os.path.basename(video_path)}_{frame_idx}.jpg"
        cv2.imwrite(
            f'/content/yawning_dataset/{label}/{filename}',
            mouth_resized
        )
    cap.release()
```

### 4.6 Veri Artırma

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,          # ±10° döndürme (ağız bölgesi daha kısıtlı)
    width_shift_range=0.10,     # ±%10 yatay öteleme
    height_shift_range=0.10,    # ±%10 dikey öteleme
    horizontal_flip=True,       # Yatay çevirme
    brightness_range=[0.8, 1.2], # Parlaklık değişimi ±%20
    zoom_range=0.10             # ±%10 zoom
)
```

Göz modeliyle karşılaştırıldığında ağız modeli için rotasyon aralığı daha küçük (±10° vs ±15°) tutulmuştur. Bunun nedeni ağız bölgesinin pozisyon değişikliklerine daha duyarlı olmasıdır.

Train/Val/Test bölünmesi (%70/%15/%15):

```python
from sklearn.model_selection import train_test_split
import shutil, random

all_images = []
for label in ['yawning', 'normal']:
    for img in os.listdir(f'/content/yawning_dataset/{label}'):
        all_images.append((f'/content/yawning_dataset/{label}/{img}', label))

random.shuffle(all_images)
train, temp = train_test_split(all_images, test_size=0.30, random_state=42)
val, test   = train_test_split(temp,       test_size=0.50, random_state=42)

# Dosyaları bölünmüş dizinlere kopyala
for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
    for img_path, label in split_data:
        dest = f'/content/yawning_split/{split_name}/{label}/'
        os.makedirs(dest, exist_ok=True)
        shutil.copy(img_path, dest)
```

```python
IMG_HEIGHT = 64
IMG_WIDTH  = 64
BATCH_SIZE = 64
EPOCHS     = 50
```

### 4.7 Model Mimarisi

Esnerme modeli göz modelinden daha küçük ama Batch Normalization içeren bir mimariye sahiptir:

```python
model = models.Sequential([
    # ── Blok 1 ──────────────────────────────────────────
    layers.Conv2D(32, (3,3), activation='relu',
                  input_shape=(64, 64, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # ── Blok 2 ──────────────────────────────────────────
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # ── Blok 3 ──────────────────────────────────────────
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # ── Blok 4 ──────────────────────────────────────────
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),   # MaxPool yerine GAP
    layers.Dropout(0.4),

    # ── Sınıflandırıcı ──────────────────────────────────
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
```

#### Mimari Detayları

| Katman | Filtre/Nöron | Aktivasyon | Çıktı Boyutu | Özellik |
|--------|-------------|------------|--------------|---------|
| Conv2D | 32 × (3×3) | ReLU | 62×62×32 | — |
| BatchNorm | — | — | 62×62×32 | Aktivasyon normaliz. |
| MaxPool | 2×2 | — | 31×31×32 | — |
| Dropout | 0.25 | — | 31×31×32 | — |
| Conv2D | 64 × (3×3) | ReLU | 29×29×64 | — |
| BatchNorm | — | — | 29×29×64 | — |
| MaxPool | 2×2 | — | 14×14×64 | — |
| Dropout | 0.25 | — | 14×14×64 | — |
| Conv2D | 128 × (3×3) | ReLU | 12×12×128 | — |
| BatchNorm | — | — | 12×12×128 | — |
| MaxPool | 2×2 | — | 6×6×128 | — |
| Dropout | 0.25 | — | 6×6×128 | — |
| Conv2D | 256 × (3×3) | ReLU | 4×4×256 | — |
| BatchNorm | — | — | 4×4×256 | — |
| GlobalAvgPool | — | — | 256 | Uzamsal bilgi özetler |
| Dropout | 0.40 | — | 256 | — |
| Dense | 128 | ReLU | 128 | — |
| Dropout | 0.30 | — | 128 | — |
| Dense | 1 | Sigmoid | 1 | İkili çıktı |

**Göz Modeliyle Karşılaştırma:**

| Özellik | Göz Modeli | Esnerme Modeli |
|---------|-----------|----------------|
| Giriş boyutu | 84×84 | 64×64 |
| Evrişim blok sayısı | 4 | 4 |
| Maksimum filtre | 512 | 256 |
| Batch Normalization | Yok | Var (her blokta) |
| Son havuzlama | MaxPooling | GlobalAveragePooling |
| Model boyutu | ~11 MB | ~1.7 MB |
| Toplam parametre | ~2.76M | ~700K |

**Batch Normalization kullanım gerekçesi:** YawDD veri seti video frame'lerinden oluştuğundan aydınlatma değişkenlikleri yüksektir. BN, aktivasyonları normalize ederek bu değişkenliğe karşı dayanıklılık sağlar ve eğitimi hızlandırır.

**GlobalAveragePooling2D kullanım gerekçesi:** MaxPooling yerine GAP kullanılması, son evrişim haritasını tam bağlantılı katmana geçmeden önce uzamsal ortalama alarak temsil etmesini sağlar. Bu yaklaşım parametre sayısını önemli ölçüde azaltır ve overfitting'e karşı korur.

### 4.8 Derleme ve Eğitim Yapılandırması

```python
mixed_precision.set_global_policy('mixed_float16')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(
        '/content/drive/MyDrive/havelsandataset/yawning_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,              # Göz modeline göre daha sabırlı
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]
```

EarlyStopping patience değeri göz modelinden (5) daha yüksek (7) tutulmuştur. Bu, esnerme modelinin eğitiminin daha dalgalı olabileceği varsayımına dayanır — çünkü veri seti video frame'lerinden elde edilmiş ve etiketler MAR tabanlıdır.

### 4.9 Eğitim Süreci

```python
EPOCHS = 50

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)
```

### 4.10 Değerlendirme

```python
# Test değerlendirmesi
test_loss, test_acc = model.evaluate(test_generator)

# Karmaşıklık matrisi ve sınıflandırma raporu
from sklearn.metrics import classification_report, confusion_matrix

y_pred = (model.predict(test_generator) > 0.5).astype(int)
y_true = test_generator.classes

print(classification_report(y_true, y_pred,
      target_names=['Normal', 'Yawning']))
```

Çıkarım demo fonksiyonu:

```python
def predict_yawning(frame_bgr, detector, model, img_size=64):
    """
    Tek frame'den esnerme tahmini.

    Args:
        frame_bgr: webcam'den gelen BGR frame
        detector:  MediaPipe FaceLandmarker nesnesi
        model:     Eğitilmiş Keras modeli
        img_size:  Model giriş boyutu (64)

    Returns:
        label:      'yawning' veya 'normal'
        confidence: Model güven değeri [0-1]
        mar:        Hesaplanan MAR değeri
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    if not result.face_landmarks:
        return None, 0.0, 0.0

    lm = result.face_landmarks[0]
    mar = compute_mar(lm, w, h)
    mouth_crop = crop_mouth_region(frame_bgr, lm, w, h)

    img = cv2.resize(mouth_crop, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)

    prob = float(model.predict(img, verbose=0)[0][0])
    label = 'yawning' if prob > 0.5 else 'normal'

    return label, prob, mar
```

### 4.11 Eşik Değerleri ve Model Kaydı

```python
import json

thresholds = {
    "mar_yawning_threshold": float(MAR_THRESHOLD),   # 0.3233
    "mar_normal_max":        float(MAR_NORMAL_MAX),  # 0.1459
    "img_size":              64,
    "class_indices":         {"normal": 0, "yawning": 1}
}

with open('yawning_thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)

# Google Drive'a kaydet
model.save('/content/drive/MyDrive/havelsandataset/yawning_model.h5')
shutil.copy('yawning_thresholds.json',
            '/content/drive/MyDrive/havelsandataset/yawning_thresholds.json')
```

---

## 5. Nihai Sistem Mimarisi

Eğitimden sonra nihai sistem dört modüle ayrılmıştır:

```
utarldd2/
├── config.py          ← Tüm sabitler ve eşikler
├── models.py          ← Model yükleme ve çıkarım fonksiyonları
├── detector.py        ← MediaPipe yüz tespiti, kırpma, head pose
├── fusion.py          ← Yorgunluk skoru hesaplama (FatigueState sınıfı)
├── main.py            ← Kamera döngüsü ve UI görselleştirme
├── models/
│   ├── mrl_eye_final_model.h5   (~11 MB)
│   ├── yawning_model.h5         (~1.7 MB)
│   └── yawning_thresholds.json
└── face_landmarker.task          (~3.7 MB, MediaPipe yüz modeli)
```

**Modüler tasarımın amacı:** Backend servisi geliştirirken `models.py`, `detector.py` ve `fusion.py` modülleri değiştirilmeden kullanılabilir; yalnızca `main.py` HTTP endpoint'leriyle değiştirilen bir servis katmanına dönüştürülür.

---

## 6. config.py

Tüm eşik değerleri, pencere boyutları, landmark indeksleri ve ağırlıklar tek bir dosyada merkezi olarak yönetilir.

```python
# Model boyutları
EYE_IMG_SIZE  = 84   # Göz modelinin eğitildiği boyut
YAWN_IMG_SIZE = 64   # Esnerme modelinin eğitildiği boyut

# PERCLOS penceresi
PERCLOS_WINDOW    = 45    # ~30fps × 1.5sn = 45 frame
EYE_CLOSED_THRESH = 0.55  # Bu eşiğin üzerindeki olasılık "kapalı" sayılır

# Microsleep tespiti
EYE_STREAK_FRAMES = 15    # ~0.5 sn üst üste kapalı → microsleep

# Esnerme
YAWN_WINDOW = 30    # 30 frame'lik hareketli ortalama (~1sn)
YAWN_THRESH = 0.55  # Yumuşatılmış skor bu eşiği aşarsa "esneme"

# Baş pozisyonu eşikleri
HEAD_PITCH_WARN  = 15.0  # derece — ilk uyarı
HEAD_PITCH_ALERT = 25.0  # derece — yüksek tehlike

# Fusion ağırlıkları (toplam = 1.0)
W_PERCLOS    = 0.40
W_HEAD_PITCH = 0.30
W_YAWN       = 0.30

# Alarm seviyeleri
SCORE_WARN  = 30   # 0–30: UYANIK, 30–60: UYARI, 60–100: UYUKLUYOR
SCORE_DROWS = 60

# EMA katsayısı (küçük = daha yavaş tepki, büyük = daha hızlı)
EMA_ALPHA = 0.15
```

**MediaPipe 478 noktalı modelde göz ve ağız landmark indeksleri:**

```python
LEFT_EYE  = [362, 385, 387, 263, 373, 380]  # Sol göz 6 noktası
RIGHT_EYE = [33,  160, 158, 133, 153, 144]  # Sağ göz 6 noktası

MOUTH_OUTLINE = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,   # Üst dudak (10 nokta)
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146  # Alt dudak (10 nokta)
]
```

**Head pose için 3D referans modeli:**

```python
HEAD_3D_POINTS = np.array([
    [  0.0,    0.0,    0.0],   # Burun ucu (landmark 1)
    [  0.0, -330.0,  -65.0],   # Çene ucu (landmark 152)
    [-225.0,  170.0, -135.0],  # Sol göz dış köşesi (landmark 263)
    [ 225.0,  170.0, -135.0],  # Sağ göz dış köşesi (landmark 33)
    [-150.0, -150.0, -125.0],  # Sol ağız köşesi (landmark 287)
    [ 150.0, -150.0, -125.0],  # Sağ ağız köşesi (landmark 57)
], dtype=np.float64)
# Birim: milimetre (ortalama yetişkin yüz geometrisi)
```

---

## 7. models.py

Model yükleme işlemi ve her iki modelin çıkarım (inference) fonksiyonlarını içerir.

```python
def _load_models():
    # 1. Mixed Precision ayarı
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    # 2. Eski modeldeki bilinmeyen config anahtarını yoksay
    #    (eğitim sırasındaki TF sürümü farklılığından kaynaklanan uyumluluk fix'i)
    _orig = tf.keras.layers.Dense.from_config.__func__
    @classmethod
    def _patched(cls, config):
        config.pop('quantization_config', None)  # Bilinmeyen anahtar kaldırılır
        return _orig(cls, config)
    tf.keras.layers.Dense.from_config = _patched

    eye  = tf.keras.models.load_model(EYE_MODEL_PATH,  compile=False)
    yawn = tf.keras.models.load_model(YAWN_MODEL_PATH, compile=False)
    return eye, yawn

# Modüle ilk kez import edildiğinde otomatik yüklenir
eye_model, yawn_model = _load_models()
```

**Çıkarım fonksiyonları:**

```python
def infer_eye(crop: np.ndarray) -> float:
    """
    BGR formatında kırpılmış göz görüntüsünden kapalı göz olasılığı.

    Pipeline:
      BGR → RGB dönüşüm
      84×84 yeniden boyutlandırma
      [0,255] → [0,1] normalizasyon
      Boyut genişletme (1, 84, 84, 3)
      Model çıkarımı
      float32'ye cast + scalar'a dönüştürme

    Returns: [0,1] arası float, 1.0 = tamamen kapalı
    """

def infer_yawn(crop: np.ndarray) -> float:
    """
    BGR formatında kırpılmış ağız görüntüsünden esnerme olasılığı.

    Pipeline: (infer_eye ile aynı, boyut 64×64)

    Returns: [0,1] arası float, 1.0 = kesinlikle esneme
    """
```

---

## 8. detector.py

MediaPipe FaceLandmarker kurulumu, bölge kırpma fonksiyonları, baş pozisyonu hesaplama ve tek frame'den tüm ham özelliklerin çıkarılması.

### MediaPipe Kurulumu

```python
_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,   # Senkron mod (video için LIVE_STREAM)
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)

face_landmarker = FaceLandmarker.create_from_options(_options)
```

`IMAGE` modu seçilmiştir çünkü ana döngü her frame'i ayrı ayrı işlemektedir. `LIVE_STREAM` modu asenkron callback gerektirir.

### Bölge Kırpma

```python
def crop_eye_region(frame, landmarks, eye_ids, w, h, padding=0.35):
    """
    6 göz landmark'ından bounding box hesaplar ve kırpar.

    padding=0.35 → bounding box her yönde %35 genişletilir
    Dikey yönde padding * 2.5 uygulanır (göz üst/alt yönde daha uzaktır)
    """
    xs = [landmarks[i].x * w for i in eye_ids]
    ys = [landmarks[i].y * h for i in eye_ids]
    pw = (max(xs) - min(xs)) * padding
    ph = (max(ys) - min(ys)) * padding
    x1 = max(0, int(min(xs) - pw))
    y1 = max(0, int(min(ys) - ph * 2.5))   # Dikey padding 2.5×
    x2 = min(w, int(max(xs) + pw))
    y2 = min(h, int(max(ys) + ph * 2.5))
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def crop_mouth(frame, landmarks, w, h, padding=0.4):
    """
    20 ağız landmark'ından bounding box hesaplar ve kırpar.
    padding=0.4 → ağız çevresinde biraz daha fazla bağlam
    """
```

### Baş Pozisyonu Hesaplama

Gerçek pitch ve yaw açıları, 2D görüntü noktaları ile bilinen 3D yüz geometrisi arasındaki perspektif dönüşümü kullanılarak hesaplanır:

```python
def get_head_pose(landmarks, w, h):
    """
    Adım 1: 6 landmark'ın görüntü koordinatları
    """
    img_pts = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in HEAD_LM_IDS],
        dtype=np.float64
    )

    """
    Adım 2: Kamera iç parametreleri (pinhole model varsayımı)
    Odak uzaklığı = görüntü genişliği (yaklaşık)
    Optik merkez = görüntü merkezi
    """
    focal = w
    cam_matrix = np.array([
        [focal,  0,    w/2],
        [0,    focal,  h/2],
        [0,      0,     1 ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))  # Distorsiyon ihmal edilir

    """
    Adım 3: PnP (Perspective-n-Point) çözümü
    3D referans noktaları + 2D görüntü noktaları → rotasyon + öteleme vektörleri
    """
    ok, rvec, _ = cv2.solvePnP(
        HEAD_3D_POINTS, img_pts, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    """
    Adım 4: Euler açılarına dönüşüm
    Rodrigues formülü: rotasyon vektörü → 3×3 rotasyon matrisi
    """
    rot_mat, _ = cv2.Rodrigues(rvec)

    pitch = np.degrees(np.arctan2(rot_mat[2,1], rot_mat[2,2]))
    yaw   = np.degrees(np.arctan2(
        -rot_mat[2,0],
        np.sqrt(rot_mat[2,1]**2 + rot_mat[2,2]**2)
    ))
    return pitch, yaw
    # Pozitif pitch → öne eğilme (baş düşüyor)
    # Negatif pitch → arkaya eğilme
    # Pozitif yaw   → sola dönme
```

### Ana Özellik Çıkarım Fonksiyonu

```python
def extract_features(frame) -> dict:
    """
    Bir BGR frame'den tüm ham özellikleri çıkarır.

    Returns:
        face_found: bool          — yüz tespit edildi mi
        eye_prob:   float [0-1]   — ortalama göz kapanma olasılığı
        yawn_prob:  float [0-1]   — esnerme olasılığı
        pitch:      float (°)     — baş öne eğilme açısı
        yaw:        float (°)     — baş yana dönme açısı
        lbox:       tuple         — sol göz bounding box (x1,y1,x2,y2)
        rbox:       tuple         — sağ göz bounding box
        mbox:       tuple         — ağız bounding box
    """
    # 1. BGR → RGB → MediaPipe Image
    # 2. FaceLandmarker.detect()
    # 3. Yüz bulunamazsa sıfır değerli dict döndür
    # 4. Göz bölgelerini kırp → infer_eye() × 2 → ortalama
    # 5. Ağız bölgesini kırp → infer_yawn()
    # 6. get_head_pose() ile pitch/yaw
    # 7. Tüm değerleri dict olarak döndür
```

---

## 9. fusion.py

Zamansal durum yönetimi ve çok sinyalli yorgunluk skoru hesaplama.

### FatigueState Sınıfı

```python
class FatigueState:
    def __init__(self):
        self._eye_closed_log  = []   # PERCLOS için kayan pencere
        self._eye_streak_log  = []   # Streak (microsleep) tespiti
        self._yawn_history    = []   # Esnerme hareketli ortalaması
        self.ema_score        = 0.0  # EMA yumuşatılmış yorgunluk skoru
```

**`update()` metodu akışı:**

```
Yeni frame verileri (eye_prob, yawn_prob, pitch)
    │
    ├─► PERCLOS hesaplama
    │     eye_closed_log'a is_closed ekle
    │     PERCLOS_WINDOW (45) aşılırsa en eskiyi çıkar
    │     perclos = mean(eye_closed_log)
    │
    ├─► Streak tespiti
    │     eye_streak_log'a is_closed ekle
    │     EYE_STREAK_FRAMES (15) aşılırsa en eskiyi çıkar
    │     eye_streak = (len==15) AND all(True)
    │
    ├─► Esnerme yumuşatma
    │     yawn_history'e yawn_prob ekle
    │     YAWN_WINDOW (30) aşılırsa en eskiyi çıkar
    │     smooth_yawn = mean(yawn_history)
    │     yawn_drowsy = smooth_yawn > 0.55
    │
    ├─► Ham skor hesaplama
    │     if eye_streak → raw_score = 100
    │     else → ağırlıklı toplam × 100
    │
    ├─► EMA yumuşatma
    │     if eye_streak → ema_score = 100
    │     else → ema_score = 0.15 × raw + 0.85 × ema_prev
    │
    └─► Seviye belirleme + dict döndür
```

**`reset()` metodu:** Backend servisinde oturum sıfırlandığında veya yeni bir sürücü başladığında tüm geçmişi temizler.

### Alt Skor Fonksiyonları

```python
def _perclos_to_score(perclos):
    # 0 → 0.40 aralığını 0 → 1.0'a doğrusal olarak eşler
    # 0.40 ve üzeri her zaman 1.0 döndürür
    return min(1.0, perclos / 0.40)

def _head_pitch_to_score(pitch_abs):
    # 15° altında sıfır
    # 15° – 25° aralığında 0 → 1.0 doğrusal artış
    # 25° üzerinde 1.0
    if pitch_abs < HEAD_PITCH_WARN:   # < 15°
        return 0.0
    return min(1.0, (pitch_abs - 15) / 10.0)

def _yawn_to_score(smooth_yawn):
    # 0.30 altında sıfır (gürültü filtresi)
    # 0.30 – 0.55 aralığında 0 → 1.0 doğrusal artış
    # 0.55 üzerinde 1.0
    return min(1.0, smooth_yawn / 0.55) if smooth_yawn > 0.3 else 0.0
```

### Seviye Bilgileri

```python
LEVEL_INFO = {
    0: ("UYANIK",          (0, 200, 80)),    # Yeşil
    1: ("UYARI",           (0, 165, 255)),   # Turuncu
    2: ("!! UYUKLUYOR !!", (0, 40, 220)),    # Kırmızı
}
```

---

## 10. main.py

Webcam döngüsü, frame işleme ve gerçek zamanlı görselleştirme.

### Ana Döngü

```python
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = FatigueState()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)           # Ayna görüntüsü

        feats = extract_features(frame)      # detector.py

        fusion_result = state.update(        # fusion.py
            feats["eye_prob"],
            feats["yawn_prob"],
            feats["pitch"]
        )

        out = render(frame, feats, fusion_result)
        cv2.imshow("Drowsiness Fusion  |  q = cikis", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

### Görselleştirme Bileşenleri

**1. Göz/Ağız Bounding Box'ları:**
- PERCLOS > %25 veya esnerme tespit edilmişse kırmızı, aksi halde yeşil
- Sol göz, sağ göz, ağız için ayrı kutucuklar

**2. Üst Durum Bandı (55px yükseklik):**
- Koyu arka plan
- Ortalanmış durum metni (UYANIK/UYARI/!! UYUKLUYOR !!)
- Duruma göre renk (yeşil/turuncu/kırmızı)

**3. Tam Genişlik Skor Çubuğu (12px, bandın hemen altında):**
- 0–100 arası ema_score'u görsel olarak gösterir
- %30 eşiğinde sarı dikey çizgi
- %60 eşiğinde kırmızı dikey çizgi

**4. Sol Bilgi Paneli (390×240px):**
```
Skor    :  XX.X / 100   [skor renginde]
PERCLOS :  XX.X%        [kırmızı/yeşil]
Streak  :  Yok / EVET   [kırmızı/yeşil]
Bas Pitch: +XX.X  Yaw: +XX.X deg  [kırmızı/yeşil]
Esname  :  XX.X%  NORMAL/ESNEME   [kırmızı/yeşil]
────────────────────────────────
[PERCLOS çubuğu] |25%      |40%
[Esnerme çubuğu] |55%
[Pitch çubuğu]   |15°   |25°
```

**5. Overlay Şeffaflığı:**
```python
frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
# %85 overlay + %15 orijinal frame = hafif şeffaf etki
```

---

## 11. Özellikler ve Çıkarım Pipeline'ı

Frame'den nihai karara giden tam pipeline:

```
Webcam (1280×720) → cv2.flip (ayna)
        │
        ▼
[BGR → RGB] → MediaPipe FaceLandmarker
        │
        ├── Yüz bulunamadı → tüm değerler 0.0
        │
        └── 478 noktalı yüz landmark'ları
                │
                ├─► Sol göz 6 nokta (362,385,387,263,373,380)
                │     → crop_eye_region() → BGR kırpı 84×84
                │     → infer_eye() → sol_prob [0-1]
                │
                ├─► Sağ göz 6 nokta (33,160,158,133,153,144)
                │     → crop_eye_region() → BGR kırpı 84×84
                │     → infer_eye() → sag_prob [0-1]
                │     eye_prob = (sol_prob + sag_prob) / 2
                │
                ├─► Ağız 20 nokta
                │     → crop_mouth() → BGR kırpı 64×64
                │     → infer_yawn() → yawn_prob [0-1]
                │
                └─► 6 anahtar nokta (1,152,263,33,287,57)
                      → get_head_pose() → cv2.solvePnP
                      → pitch, yaw (derece)

FatigueState.update(eye_prob, yawn_prob, pitch)
        │
        ├─► is_closed = eye_prob > 0.55
        │
        ├─► PERCLOS penceresi (45 frame)
        │     perclos = fraction(closed_frames / 45)
        │
        ├─► Streak penceresi (15 frame)
        │     eye_streak = ALL 15 frames closed?
        │
        ├─► Esnerme penceresi (30 frame)
        │     smooth_yawn = mean(last 30 yawn_probs)
        │     yawn_drowsy = smooth_yawn > 0.55
        │
        ├─► Ham skor
        │     if eye_streak → 100.0
        │     else:
        │       s = 0.40 × PERCLOS_score
        │         + 0.30 × HEAD_PITCH_score
        │         + 0.30 × YAWN_score
        │       raw = s × 100
        │
        └─► EMA yumuşatma
              ema = 0.15 × raw + 0.85 × ema_prev
              level = 0/1/2 → UYANIK/UYARI/UYUKLUYOR

render() → OpenCV overlay → imshow()
```

---

## 12. Yorgunluk Skoru Matematiksel Modeli

### Alt Skor Fonksiyonları

**PERCLOS skoru:**

```
perclos_score(p) = min(1, p / 0.40)

p = 0.00 → score = 0.00  (hiç kapanma yok)
p = 0.20 → score = 0.50
p = 0.40 → score = 1.00  (tam skor)
p = 0.80 → score = 1.00  (üst sınır)
```

**Baş pitch skoru:**

```
pitch_score(θ) = 0,                           θ < 15°
               = min(1, (θ - 15) / 10),    15° ≤ θ ≤ 25°
               = 1,                           θ > 25°

θ = 10° → score = 0.00
θ = 15° → score = 0.00  (eşik başlangıcı)
θ = 20° → score = 0.50
θ = 25° → score = 1.00
θ = 35° → score = 1.00
```

**Esnerme skoru:**

```
yawn_score(y) = 0,                     y ≤ 0.30
              = min(1, y / 0.55),   0.30 < y ≤ 0.55
              = 1,                     y > 0.55

y = 0.20 → score = 0.00  (gürültü filtresi)
y = 0.30 → score = 0.55
y = 0.45 → score = 0.82
y = 0.55 → score = 1.00
```

### Birleşik Ham Skor

```
raw_score = (0.40 × perclos_score
           + 0.30 × pitch_score
           + 0.30 × yawn_score) × 100

Aralık: [0, 100]
```

**Örnekler:**

| Senaryo | PERCLOS | Pitch | Yawn | Ham Skor |
|---------|---------|-------|------|----------|
| Tamamen uyanık | 0% | 5° | 0.1 | 0.0 |
| Hafif yorgun | 15% | 12° | 0.2 | 15.0 |
| Orta yorgun | 25% | 18° | 0.4 | 40.3 |
| Ciddi yorgun | 40% | 25° | 0.6 | 79.1 |
| Microsleep | — | — | — | 100.0 |

### EMA Yumuşatma

Anlık gürültüyü bastırmak için Üstel Hareketli Ortalama uygulanır:

```
ema_t = α × raw_t + (1-α) × ema_{t-1}
α = 0.15

Zaman sabiti (yaklaşık): τ = 1/α = 6.7 frame ≈ 0.22 sn @ 30fps
```

Bu ayar:
- Ani yanlış pozitifleri (tek frame gürültüsü) önler
- Gerçek yorgunluk sinyallerine ~1–2 saniye içinde tepki verir
- Microsleep durumunda EMA bypass edilir ve anında 100 atanır

### Alarm Seviyeleri

```
ema_score ∈ [ 0, 30) → Level 0: UYANIK        (yeşil)
ema_score ∈ [30, 60) → Level 1: UYARI          (turuncu)
ema_score ∈ [60,100] → Level 2: !! UYUKLUYOR ! (kırmızı)
```

---

## 13. Kullanılan Teknolojiler

| Kütüphane | Sürüm | Kullanım Amacı |
|-----------|-------|----------------|
| TensorFlow/Keras | 2.19.0 | CNN model eğitimi ve çıkarım |
| MediaPipe | — | 478 noktalı yüz landmark tespiti |
| OpenCV (cv2) | — | Video yakalama, görüntü işleme, UI |
| NumPy | — | Sayısal hesaplamalar, matris işlemleri |
| scikit-learn | — | Veri bölme, metrik hesaplama |
| matplotlib/seaborn | — | Eğitim görselleştirmesi, karmaşıklık matrisi |
| Google Colab | — | Eğitim ortamı (GPU destekli) |
| Kaggle API | — | MRL Eye Dataset indirme |

**Donanım Ortamı (Eğitim):**
- Google Colab ücretsiz tier (Tesla T4 veya K80 GPU)
- Mixed Precision (float16) ile GPU bellek optimizasyonu

**Donanım Ortamı (Çıkarım):**
- CPU tabanlı gerçek zamanlı çalışma (GPU gerekmez)
- 1280×720 webcam akışı, ~30fps hedef

---

*Rapor Tarihi: Mart 2026*
*Sistem Sürümü: 1.0 — Modüler mimari (config / models / detector / fusion / main)*
