# Sürücü Yorgunluğu Tespiti — Proje Özeti

## Ne Yapıyor?

Webcam'den gerçek zamanlı video alarak sürücünün yorgun/uykulu olup olmadığını tespit eder. İki CNN modeli + MediaPipe yüz landmark tespiti kullanır. Çıktı 0–100 arası bir yorgunluk skoru ve üç seviyeli bir alarmdir: **UYANIK → UYARI → UYUKLUYOR**.

---

## Sistem Nasıl Çalışır?

Her frame için üç sinyal hesaplanır ve ağırlıklı olarak birleştirilir:

```
Webcam frame
    ↓
MediaPipe (478 noktalı yüz)
    ├── Göz bölgesi → CNN → göz kapanma olasılığı [0-1]
    ├── Ağız bölgesi → CNN → esnerme olasılığı [0-1]
    └── 6 anahtar nokta → solvePnP → baş pitch açısı (°)
         ↓
   Ağırlıklı Skor (PERCLOS×0.40 + Pitch×0.30 + Yawn×0.30)
         ↓
   EMA Yumuşatma (α=0.15)
         ↓
   0-30: UYANIK | 30-60: UYARI | 60-100: UYUKLUYOR
```

---

## Modeller

### Model 1 — Göz Durumu (mrl_eye_final_model.h5)
- **Veri:** MRL Eye Dataset — 84.898 kızılötesi göz görüntüsü, 37 denek, dengeli sınıf
- **Mimari:** 4 Conv blok (64→128→256→512 filtre) + 2 Dense katman, Sigmoid çıkış
- **Giriş:** 84×84 RGB
- **Eğitim:** 100 epoch, batch=128, Adam lr=0.001, mixed float16

### Model 2 — Esnerme (yawning_model.h5)
- **Veri:** YawDD — 29 araç içi video; frame'ler **MAR (Ağız Açıklık Oranı)** ile otomatik etiketlendi
  - MAR ≥ 0.3233 → esneme | MAR ≤ 0.1459 → normal | arası dışlandı
- **Mimari:** 4 Conv blok (32→64→128→256) + BatchNorm + GlobalAvgPooling, Sigmoid çıkış
- **Giriş:** 64×64 RGB
- **Eğitim:** 50 epoch, batch=64, Adam lr=0.001, mixed float16

---

## Önemli Metrikler

| Sinyal | Yöntem | Eşik/Pencere |
|--------|--------|-------------|
| PERCLOS | Son 45 frame'de kapalı göz oranı | >%40 → tam skor |
| Microsleep (Streak) | 15 frame üst üste kapalı | Anında skor=100 |
| Esnerme | 30 frame hareketli ortalama | >0.55 → tam skor |
| Baş Pitch | solvePnP ile açı hesabı | 15°–25° arası kademeli |

---

## Dosya Yapısı

```
config.py    ← tüm eşikler ve sabitler
models.py    ← model yükleme + infer_eye / infer_yawn
detector.py  ← MediaPipe + bölge kırpma + head pose → extract_features()
fusion.py    ← FatigueState sınıfı → ağırlıklı skor + EMA
main.py      ← kamera döngüsü + OpenCV UI
```

`FatigueState.update(eye_prob, yawn_prob, pitch)` tek çağrısı tüm zaman serisi mantığını yönetir; backend servisi eklenirken sadece `main.py` değiştirilecek.

---

## Tasarım Kararları

- **Neden MAR tabanlı otomatik etiketleme?** YawDD videolarında insan etiketlemesi yoktur; istatistiksel persentil eşikleriyle (82. ve 40.) belirsiz kareler atlanarak temiz veri elde edildi.
- **Neden EMA?** Anlık model gürültüsünü bastırır; gerçek yorgunluk sinyaline ~1–2 sn içinde tepki verir. Microsleep'te bypass edilir, anında 100 atanır.
- **Neden PERCLOS ağırlığı en yüksek (%40)?** Yorgunluk literatüründe en güvenilir ve kanıtlanmış gösterge olduğu için.
- **Neden GlobalAveragePooling (esnerme modeli)?** Parametre sayısını düşürür, video veri setlerinden gelen aydınlatma değişkenliğine karşı dayanıklıdır.
