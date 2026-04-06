import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# ── Ayarlar ──────────────────────────────────────────────────────────────────
EYE_MODEL_PATH  = r"models/mrl_eye_final_model.h5"
YAWN_MODEL_PATH = r"models/yawning_model.h5"
FACE_MODEL_PATH = r"face_landmarker.task"

EYE_IMG_SIZE  = 84
YAWN_IMG_SIZE = 64

# PERCLOS (~30fps → 45 frame ≈ 1.5 sn)
PERCLOS_WINDOW    = 45
EYE_CLOSED_THRESH = 0.55   # tek frame'de "göz kapalı" sayılma eşiği

# Streak: üst üste bu kadar frame kapalıysa anında tetikle (~0.5 sn)
EYE_STREAK_FRAMES = 15

# Esneme
YAWN_WINDOW = 30
YAWN_THRESH = 0.55

# Head Pose
HEAD_PITCH_WARN  = 15.0  # derece — öne eğilme uyarı eşiği
HEAD_PITCH_ALERT = 25.0  # derece — güçlü eşik

# Fusion ağırlıkları
W_PERCLOS    = 0.40
W_HEAD_PITCH = 0.30
W_YAWN       = 0.30

# Skor seviyeleri
SCORE_WARN  = 30   # UYARI
SCORE_DROWS = 60   # UYUKLUYOR

# EMA yumuşatma katsayısı
EMA_ALPHA = 0.15

# ── Landmark indeksleri ───────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158,  133, 153, 144]

MOUTH_OUTLINE = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                 375, 321, 405, 314, 17, 84, 181, 91, 146]

# Head pose için 3D referans modeli (gerçek yüz geometrisi, mm)
HEAD_3D_POINTS = np.array([
    [0.0,    0.0,    0.0   ],   # Burun ucu (1)
    [0.0,   -330.0, -65.0  ],   # Çene (152)
    [-225.0, 170.0, -135.0 ],   # Sol göz dış köşe (263)
    [225.0,  170.0, -135.0 ],   # Sağ göz dış köşe (33)
    [-150.0,-150.0, -125.0 ],   # Sol ağız köşe (287)
    [150.0, -150.0, -125.0 ],   # Sağ ağız köşe (57)
], dtype=np.float64)

HEAD_LM_IDS = [1, 152, 263, 33, 287, 57]

# ── Model Yükle ───────────────────────────────────────────────────────────────
print("Modeller yukleniyor...")

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

_orig_dense_from_config = tf.keras.layers.Dense.from_config.__func__
@classmethod
def _patched_dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return _orig_dense_from_config(cls, config)
tf.keras.layers.Dense.from_config = _patched_dense_from_config

eye_model  = tf.keras.models.load_model(EYE_MODEL_PATH,  compile=False)
yawn_model = tf.keras.models.load_model(YAWN_MODEL_PATH, compile=False)
print("Modeller hazir.")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)

# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────────
def crop_eye_region(frame, landmarks, eye_ids, w, h, padding=0.35):
    xs = [landmarks[i].x * w for i in eye_ids]
    ys = [landmarks[i].y * h for i in eye_ids]
    pw = (max(xs) - min(xs)) * padding
    ph = (max(ys) - min(ys)) * padding
    x1 = max(0, int(min(xs) - pw))
    y1 = max(0, int(min(ys) - ph * 2.5))
    x2 = min(w, int(max(xs) + pw))
    y2 = min(h, int(max(ys) + ph * 2.5))
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def crop_mouth(frame, landmarks, w, h, padding=0.4):
    xs = [landmarks[i].x * w for i in MOUTH_OUTLINE]
    ys = [landmarks[i].y * h for i in MOUTH_OUTLINE]
    pw = (max(xs) - min(xs)) * padding
    ph = (max(ys) - min(ys)) * padding
    x1 = max(0, int(min(xs) - pw))
    y1 = max(0, int(min(ys) - ph))
    x2 = min(w, int(max(xs) + pw))
    y2 = min(h, int(max(ys) + ph))
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def infer_eye(crop):
    if crop.size == 0:
        return 0.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (EYE_IMG_SIZE, EYE_IMG_SIZE)).astype(np.float32) / 255.0
    raw = eye_model(np.expand_dims(img, 0), training=False)
    return float(tf.cast(raw, tf.float32).numpy().flatten()[0])

def infer_yawn(crop):
    if crop.size == 0:
        return 0.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (YAWN_IMG_SIZE, YAWN_IMG_SIZE)).astype(np.float32) / 255.0
    raw = yawn_model(np.expand_dims(img, 0), training=False)
    return float(tf.cast(raw, tf.float32).numpy().flatten()[0])

def get_head_pose(landmarks, w, h):
    """
    MediaPipe landmark'larından pitch (öne-arkaya eğilme) ve yaw (yana dönme)
    açılarını döndürür (derece).
    Pozitif pitch → kamera bazında aşağı bakma (öne eğilme).
    """
    img_pts = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in HEAD_LM_IDS],
        dtype=np.float64
    )
    focal = w  # yaklaşık odak uzaklığı
    cam_matrix = np.array([
        [focal, 0,     w / 2],
        [0,     focal, h / 2],
        [0,     0,     1    ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        HEAD_3D_POINTS, img_pts, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rvec)
    # Euler açıları: pitch=x, yaw=y
    pitch = np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
    yaw   = np.degrees(np.arctan2(-rot_mat[2, 0],
                                   np.sqrt(rot_mat[2, 1]**2 + rot_mat[2, 2]**2)))
    return pitch, yaw

# ── Skor Hesaplama ────────────────────────────────────────────────────────────
def perclos_to_score(perclos):
    """PERCLOS → 0-100 skor (0.40+ = tam skor)"""
    return min(1.0, perclos / 0.40)

def head_pitch_to_score(pitch_abs):
    """Mutlak pitch açısı → 0-100 skor"""
    if pitch_abs < HEAD_PITCH_WARN:
        return 0.0
    return min(1.0, (pitch_abs - HEAD_PITCH_WARN) / (HEAD_PITCH_ALERT - HEAD_PITCH_WARN))

def yawn_to_score(smooth_yawn):
    """Esneme olasılığı → 0-100 skor"""
    return min(1.0, smooth_yawn / YAWN_THRESH) if smooth_yawn > 0.3 else 0.0

def compute_fatigue_score(perclos, pitch_abs, smooth_yawn, eye_streak):
    """Ağırlıklı yorgunluk skoru [0-100]. Streak anında maksimum skor."""
    if eye_streak:
        return 100.0

    s = (W_PERCLOS    * perclos_to_score(perclos)
       + W_HEAD_PITCH * head_pitch_to_score(pitch_abs)
       + W_YAWN       * yawn_to_score(smooth_yawn))
    return s * 100.0

def score_to_level(score):
    """Skoru 3 seviyeli alerte dönüştür."""
    if score >= SCORE_DROWS:
        return 2, "!! UYUKLUYOR !!", (0, 40, 220)
    if score >= SCORE_WARN:
        return 1, "UYARI",           (0, 165, 255)
    return 0, "UYANIK",              (0, 200, 80)

# ── Ana Döngü ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

eye_closed_log  = []
eye_streak_log  = []
yawn_history    = []

# EMA skor
ema_score = 0.0

print("Kamera acildi. Cikis icin 'q' tusuna basin.")

with FaceLandmarker.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        eye_prob   = 0.0
        yawn_prob  = 0.0
        face_found = False
        pitch      = 0.0
        yaw        = 0.0
        lbox = rbox = mbox = (0, 0, 0, 0)

        if result.face_landmarks:
            face_found = True
            lm = result.face_landmarks[0]

            left_crop,  lbox = crop_eye_region(frame, lm, LEFT_EYE,  w, h)
            right_crop, rbox = crop_eye_region(frame, lm, RIGHT_EYE, w, h)
            eye_prob = (infer_eye(left_crop) + infer_eye(right_crop)) / 2.0

            mouth_crop, mbox = crop_mouth(frame, lm, w, h)
            yawn_prob = infer_yawn(mouth_crop)

            pitch, yaw = get_head_pose(lm, w, h)

        # PERCLOS + streak
        is_closed = eye_prob > EYE_CLOSED_THRESH
        eye_closed_log.append(is_closed)
        eye_streak_log.append(is_closed)
        if len(eye_closed_log) > PERCLOS_WINDOW:
            eye_closed_log.pop(0)
        if len(eye_streak_log) > EYE_STREAK_FRAMES:
            eye_streak_log.pop(0)
        perclos    = np.mean(eye_closed_log)
        eye_streak = (len(eye_streak_log) == EYE_STREAK_FRAMES and all(eye_streak_log))

        # Esneme
        yawn_history.append(yawn_prob)
        if len(yawn_history) > YAWN_WINDOW:
            yawn_history.pop(0)
        smooth_yawn = np.mean(yawn_history)
        yawn_drowsy = smooth_yawn > YAWN_THRESH

        # Yorgunluk skoru (ham)
        pitch_abs = abs(pitch)
        raw_score = compute_fatigue_score(perclos, pitch_abs, smooth_yawn, eye_streak)

        # EMA yumuşatma (streak'te bypass)
        if eye_streak:
            ema_score = 100.0
        else:
            ema_score = EMA_ALPHA * raw_score + (1 - EMA_ALPHA) * ema_score

        level, state_text, state_color = score_to_level(ema_score)

        # ── Overlay ──────────────────────────────────────────────────────────
        overlay = frame.copy()

        # Göz/ağız kutucukları
        if face_found:
            eye_col  = (0, 80, 255) if perclos > 0.25 else (0, 220, 100)
            yawn_col = (0, 80, 255) if yawn_drowsy     else (0, 220, 100)
            cv2.rectangle(overlay, (lbox[0], lbox[1]), (lbox[2], lbox[3]), eye_col, 1)
            cv2.rectangle(overlay, (rbox[0], rbox[1]), (rbox[2], rbox[3]), eye_col, 1)
            cv2.rectangle(overlay, (mbox[0], mbox[1]), (mbox[2], mbox[3]), yawn_col, 2)

        # Üst durum bandı
        cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
        ts = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)[0]
        cv2.putText(overlay, state_text, ((w - ts[0]) // 2, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, state_color, 2)

        # Skor çubuğu — üst bandın hemen altında, tam genişlik
        bar_y  = 55
        bar_h  = 12
        bar_fill = int(w * ema_score / 100.0)
        cv2.rectangle(overlay, (0, bar_y), (w, bar_y + bar_h), (30, 30, 30), -1)
        bar_col = state_color
        if bar_fill > 0:
            cv2.rectangle(overlay, (0, bar_y), (bar_fill, bar_y + bar_h), bar_col, -1)
        # Eşik çizgileri
        for pct, col in [(SCORE_WARN / 100, (0, 200, 200)), (SCORE_DROWS / 100, (0, 80, 255))]:
            mx = int(w * pct)
            cv2.line(overlay, (mx, bar_y), (mx, bar_y + bar_h), col, 1)

        # Sol panel (genişletildi)
        px, py   = 10, 73
        pw_panel = 390
        ph_panel = 240
        cv2.rectangle(overlay, (px, py), (px + pw_panel, py + ph_panel), (18, 18, 18), -1)
        cv2.rectangle(overlay, (px, py), (px + pw_panel, py + ph_panel), (60, 60, 60),  1)

        def lbl(text, vy, color=(210, 210, 210)):
            cv2.putText(overlay, text, (px + 12, vy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1)

        eye_c    = (0, 80, 255) if perclos > 0.25  else (0, 200, 80)
        yawn_c   = (0, 80, 255) if yawn_drowsy      else (0, 200, 80)
        streak_c = (0, 80, 255) if eye_streak        else (0, 200, 80)
        pitch_c  = (0, 80, 255) if pitch_abs > HEAD_PITCH_WARN else (0, 200, 80)
        score_c  = state_color

        lbl(f"Skor    : {ema_score:5.1f} / 100", py + 28, score_c)
        lbl(f"PERCLOS : {perclos * 100:5.1f}%",  py + 53, eye_c)
        lbl(f"Streak  : {'EVET (' + str(EYE_STREAK_FRAMES) + ' frame)' if eye_streak else 'Yok'}",
            py + 78, streak_c)
        lbl(f"Bas Pitch: {pitch:+5.1f}  Yaw: {yaw:+5.1f} deg",
            py + 103, pitch_c)
        lbl(f"Esname  : {smooth_yawn * 100:5.1f}%  {'ESNEME' if yawn_drowsy else 'NORMAL'}",
            py + 128, yawn_c)

        cv2.line(overlay, (px + 10, py + 143), (px + pw_panel - 10, py + 143), (60, 60, 60), 1)

        # PERCLOS çubuğu
        def draw_bar(bar_x, bar_y_off, bar_w, bar_h_,
                     value, max_val, fill_col, label, thresholds=()):
            bx = bar_x
            by = bar_y_off
            cv2.rectangle(overlay, (bx, by), (bx + bar_w, by + bar_h_), (40, 40, 40), -1)
            fill = int(bar_w * min(value / max_val, 1.0))
            if fill > 0:
                cv2.rectangle(overlay, (bx, by), (bx + fill, by + bar_h_), fill_col, -1)
            cv2.rectangle(overlay, (bx, by), (bx + bar_w, by + bar_h_), (80, 80, 80), 1)
            for pct, tcol in thresholds:
                mx = bx + int(bar_w * pct)
                cv2.line(overlay, (mx, by - 2), (mx, by + bar_h_ + 2), tcol, 1)
            cv2.putText(overlay, label, (bx + 4, by + bar_h_ - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

        bx0 = px + 10
        bw0 = pw_panel - 20
        bh0 = 16

        pcol = (0, 40, 220) if perclos > 0.40 else ((0, 160, 220) if perclos > 0.25 else (0, 180, 80))
        draw_bar(bx0, py + 153, bw0, bh0, perclos, 1.0, pcol,
                 "PERCLOS  |25%      |40%",
                 [(0.25, (200, 200, 0)), (0.40, (0, 100, 255))])

        ycol = (0, 40, 220) if yawn_drowsy else (0, 180, 80)
        draw_bar(bx0, py + 181, bw0, bh0, smooth_yawn, 1.0, ycol,
                 f"Esname olasiligi  |{int(YAWN_THRESH * 100)}%",
                 [(YAWN_THRESH, (200, 200, 0))])

        pitch_norm = min(abs(pitch) / 40.0, 1.0)
        pcol2 = (0, 40, 220) if pitch_abs > HEAD_PITCH_ALERT else (
                 (0, 165, 255) if pitch_abs > HEAD_PITCH_WARN else (0, 180, 80))
        draw_bar(bx0, py + 209, bw0, bh0, pitch_norm, 1.0, pcol2,
                 f"Bas pitch  |{int(HEAD_PITCH_WARN)}d   |{int(HEAD_PITCH_ALERT)}d",
                 [(HEAD_PITCH_WARN / 40, (200, 200, 0)),
                  (HEAD_PITCH_ALERT / 40, (0, 100, 255))])

        # Yüz bulunamadı
        if not face_found:
            cv2.putText(overlay, "YUZ BULUNAMADI", (w // 2 - 130, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 2)

        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        cv2.imshow("Drowsiness Fusion  |  q = cikis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
