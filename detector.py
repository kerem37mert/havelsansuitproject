import numpy as np
import cv2
import mediapipe as mp
from config import (FACE_MODEL_PATH, LEFT_EYE, RIGHT_EYE, MOUTH_OUTLINE,
                    HEAD_3D_POINTS, HEAD_LM_IDS)


# ── MediaPipe kurulumu ────────────────────────────────────────────────────────
_BaseOptions           = mp.tasks.BaseOptions
_FaceLandmarker        = mp.tasks.vision.FaceLandmarker
_FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
_VisionRunningMode     = mp.tasks.vision.RunningMode

_options = _FaceLandmarkerOptions(
    base_options=_BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=_VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)

face_landmarker = _FaceLandmarker.create_from_options(_options)


# ── Kırpma fonksiyonları ──────────────────────────────────────────────────────
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


# ── Baş pozisyonu ─────────────────────────────────────────────────────────────
def get_head_pose(landmarks, w, h):
    """
    Pitch (öne-arkaya) ve yaw (yana) açılarını derece cinsinden döndürür.
    Pozitif pitch → öne eğilme.
    """
    img_pts = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in HEAD_LM_IDS],
        dtype=np.float64
    )
    focal = w
    cam_matrix = np.array([
        [focal, 0,     w / 2],
        [0,     focal, h / 2],
        [0,     0,     1    ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(
        HEAD_3D_POINTS, img_pts, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rvec)
    pitch = np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
    yaw   = np.degrees(np.arctan2(-rot_mat[2, 0],
                                   np.sqrt(rot_mat[2, 1]**2 + rot_mat[2, 2]**2)))
    return pitch, yaw


# ── Tek frame'den tüm ham özellikleri çıkar ───────────────────────────────────
def extract_features(frame):
    """
    BGR frame alır; yüz bulunursa ham özellikleri döndürür.

    Returns:
        dict ile:
          face_found (bool)
          eye_prob   (float)
          yawn_prob  (float)
          pitch, yaw (float, derece)
          lbox, rbox, mbox (tuple) — görselleştirme için bounding box'lar
    """
    from models import infer_eye, infer_yawn

    h, w = frame.shape[:2]
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {
            "face_found": False,
            "eye_prob": 0.0, "yawn_prob": 0.0,
            "pitch": 0.0, "yaw": 0.0,
            "lbox": (0,0,0,0), "rbox": (0,0,0,0), "mbox": (0,0,0,0),
        }

    lm = result.face_landmarks[0]

    left_crop,  lbox = crop_eye_region(frame, lm, LEFT_EYE,  w, h)
    right_crop, rbox = crop_eye_region(frame, lm, RIGHT_EYE, w, h)
    eye_prob = (infer_eye(left_crop) + infer_eye(right_crop)) / 2.0

    mouth_crop, mbox = crop_mouth(frame, lm, w, h)
    yawn_prob = infer_yawn(mouth_crop)

    pitch, yaw = get_head_pose(lm, w, h)

    return {
        "face_found": True,
        "eye_prob": eye_prob, "yawn_prob": yawn_prob,
        "pitch": pitch, "yaw": yaw,
        "lbox": lbox, "rbox": rbox, "mbox": mbox,
    }
