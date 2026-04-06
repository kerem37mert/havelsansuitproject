import numpy as np

# ── Model dosya yolları ───────────────────────────────────────────────────────
EYE_MODEL_PATH  = r"models/mrl_eye_final_model.h5"
YAWN_MODEL_PATH = r"models/yawning_model.h5"
FACE_MODEL_PATH = r"face_landmarker.task"

# ── Model giriş boyutları ─────────────────────────────────────────────────────
EYE_IMG_SIZE  = 84
YAWN_IMG_SIZE = 64

# ── PERCLOS ───────────────────────────────────────────────────────────────────
PERCLOS_WINDOW    = 45     # ~1.5 sn @ 30fps
EYE_CLOSED_THRESH = 0.55   # tek frame'de "göz kapalı" eşiği

# ── Streak (microsleep) ───────────────────────────────────────────────────────
EYE_STREAK_FRAMES = 15     # ~0.5 sn üst üste kapalı → anında tetikle

# ── Esneme ────────────────────────────────────────────────────────────────────
YAWN_WINDOW = 30
YAWN_THRESH = 0.55

# ── Baş pozisyonu ─────────────────────────────────────────────────────────────
HEAD_PITCH_WARN  = 15.0    # derece — uyarı eşiği
HEAD_PITCH_ALERT = 25.0    # derece — güçlü eşik

# ── Fusion ağırlıkları ────────────────────────────────────────────────────────
W_PERCLOS    = 0.40
W_HEAD_PITCH = 0.30
W_YAWN       = 0.30

# ── Skor seviyeleri ───────────────────────────────────────────────────────────
SCORE_WARN  = 30   # UYARI
SCORE_DROWS = 60   # UYUKLUYOR

# ── EMA yumuşatma katsayısı ───────────────────────────────────────────────────
EMA_ALPHA = 0.15

# ── Landmark indeksleri ───────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158,  133, 153, 144]

MOUTH_OUTLINE = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                 375, 321, 405, 314, 17, 84, 181, 91, 146]

# ── Head pose 3D referans noktaları (gerçek yüz geometrisi, mm) ───────────────
HEAD_3D_POINTS = np.array([
    [0.0,    0.0,    0.0   ],   # Burun ucu (1)
    [0.0,   -330.0, -65.0  ],   # Çene (152)
    [-225.0, 170.0, -135.0 ],   # Sol göz dış köşe (263)
    [225.0,  170.0, -135.0 ],   # Sağ göz dış köşe (33)
    [-150.0,-150.0, -125.0 ],   # Sol ağız köşe (287)
    [150.0, -150.0, -125.0 ],   # Sağ ağız köşe (57)
], dtype=np.float64)

HEAD_LM_IDS = [1, 152, 263, 33, 287, 57]
