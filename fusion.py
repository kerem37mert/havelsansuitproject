import numpy as np
from config import (
    PERCLOS_WINDOW, EYE_CLOSED_THRESH, EYE_STREAK_FRAMES,
    YAWN_WINDOW, YAWN_THRESH,
    HEAD_PITCH_WARN, HEAD_PITCH_ALERT,
    W_PERCLOS, W_HEAD_PITCH, W_YAWN,
    SCORE_WARN, SCORE_DROWS, EMA_ALPHA,
)


class FatigueState:
    """
    Frame-by-frame yorgunluk durumunu tutan sınıf.
    Backend'de de doğrudan kullanılabilir:
        state = FatigueState()
        result = state.update(eye_prob, yawn_prob, pitch)
    """

    def __init__(self):
        self._eye_closed_log  = []
        self._eye_streak_log  = []
        self._yawn_history    = []
        self.ema_score        = 0.0

    def update(self, eye_prob: float, yawn_prob: float, pitch: float) -> dict:
        """
        Yeni frame verisiyle durumu günceller.

        Args:
            eye_prob:  göz kapanma olasılığı [0-1]
            yawn_prob: esneme olasılığı [0-1]
            pitch:     baş pitch açısı (derece)

        Returns:
            dict:
              perclos     float  — kayan pencere göz kapanma oranı
              eye_streak  bool   — microsleep tetiklendi mi
              smooth_yawn float  — yumuşatılmış esneme skoru
              yawn_drowsy bool   — esneme eşiği aşıldı mı
              pitch_abs   float  — mutlak pitch açısı
              raw_score   float  — anlık ham skor [0-100]
              ema_score   float  — EMA yumuşatılmış skor [0-100]
              level       int    — 0=UYANIK, 1=UYARI, 2=UYUKLUYOR
        """
        is_closed = eye_prob > EYE_CLOSED_THRESH

        self._eye_closed_log.append(is_closed)
        self._eye_streak_log.append(is_closed)
        if len(self._eye_closed_log) > PERCLOS_WINDOW:
            self._eye_closed_log.pop(0)
        if len(self._eye_streak_log) > EYE_STREAK_FRAMES:
            self._eye_streak_log.pop(0)

        perclos    = float(np.mean(self._eye_closed_log))
        eye_streak = (len(self._eye_streak_log) == EYE_STREAK_FRAMES
                      and all(self._eye_streak_log))

        self._yawn_history.append(yawn_prob)
        if len(self._yawn_history) > YAWN_WINDOW:
            self._yawn_history.pop(0)
        smooth_yawn = float(np.mean(self._yawn_history))
        yawn_drowsy = smooth_yawn > YAWN_THRESH

        pitch_abs = abs(pitch)
        raw_score = _compute_raw_score(perclos, pitch_abs, smooth_yawn, eye_streak)

        if eye_streak:
            self.ema_score = 100.0
        else:
            self.ema_score = EMA_ALPHA * raw_score + (1 - EMA_ALPHA) * self.ema_score

        return {
            "perclos":     perclos,
            "eye_streak":  eye_streak,
            "smooth_yawn": smooth_yawn,
            "yawn_drowsy": yawn_drowsy,
            "pitch_abs":   pitch_abs,
            "raw_score":   raw_score,
            "ema_score":   self.ema_score,
            "level":       _score_to_level(self.ema_score),
        }

    def reset(self):
        self._eye_closed_log.clear()
        self._eye_streak_log.clear()
        self._yawn_history.clear()
        self.ema_score = 0.0


# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _perclos_to_score(perclos):
    return min(1.0, perclos / 0.40)

def _head_pitch_to_score(pitch_abs):
    if pitch_abs < HEAD_PITCH_WARN:
        return 0.0
    return min(1.0, (pitch_abs - HEAD_PITCH_WARN) / (HEAD_PITCH_ALERT - HEAD_PITCH_WARN))

def _yawn_to_score(smooth_yawn):
    return min(1.0, smooth_yawn / YAWN_THRESH) if smooth_yawn > 0.3 else 0.0

def _compute_raw_score(perclos, pitch_abs, smooth_yawn, eye_streak):
    if eye_streak:
        return 100.0
    s = (W_PERCLOS    * _perclos_to_score(perclos)
       + W_HEAD_PITCH * _head_pitch_to_score(pitch_abs)
       + W_YAWN       * _yawn_to_score(smooth_yawn))
    return s * 100.0

def _score_to_level(score):
    if score >= SCORE_DROWS:
        return 2
    if score >= SCORE_WARN:
        return 1
    return 0

LEVEL_INFO = {
    0: ("UYANIK",        (0, 200, 80)),
    1: ("UYARI",         (0, 165, 255)),
    2: ("!! UYUKLUYOR !!", (0, 40, 220)),
}
