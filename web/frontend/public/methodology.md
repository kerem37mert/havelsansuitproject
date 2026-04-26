# Yorgunluk Tespit Yöntemi

Sistem 3 ayrı sinyal ölçer, bunları birleştirerek 0–100 arası tek bir skor üretir.

---

## Sinyaller

### 1. PERCLOS (%40 ağırlık)
Son 45 frame'de gözün kapalı kaldığı sürenin oranı.  
Her frame'de göz CNN modeli kapanma olasılığı döndürür. `> 0.55` ise o frame "kapalı" sayılır.  
Örnek: 45 frame'in 18'i kapalıysa `PERCLOS = %40`.

### 2. Esneme (%30 ağırlık)
Ağız CNN modeli esneme olasılığı döndürür.  
Son 30 frame'in ortalaması alınır. Ortalama `> 0.55` ise "esneme var" denir.

### 3. Baş Pitch Açısı (%30 ağırlık)
Yüz landmark'larından `solvePnP` ile başın öne eğilme açısı hesaplanır.  
- `< 15°` → etkisiz  
- `15°–25°` → kademeli etki  
- `> 25°` → maksimum etki

---

## Birleştirme (Fusion)

3 sinyal normalize edilerek toplanır:

```
ham_skor = 0.40 × perclos + 0.30 × pitch + 0.30 × esneme
```

Ardından **EMA** ile yumuşatılır (`α = 0.15`):

```
ema_skor = 0.15 × ham_skor + 0.85 × önceki_skor
```

Bu sayede anlık değişimler skoru aniden etkilemez.

---

## Microsleep (Özel Durum)

15 frame üst üste (`~0.5 sn`) göz kapalıysa skor direkt **100**'e zıplar.  
Fusion hesabı atlanır — bu gerçek bir anlık uyuma anıdır.

---

## Seviyeler

| Skor | Seviye |
|------|--------|
| 0 – 30 | Uyanık |
| 30 – 60 | Uyarı |
| 60 – 100 | Uyukluyor |
