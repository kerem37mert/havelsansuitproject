import cv2
import numpy as np

from config import (EYE_STREAK_FRAMES, YAWN_THRESH, HEAD_PITCH_WARN,
                    HEAD_PITCH_ALERT, SCORE_WARN, SCORE_DROWS)
from detector import extract_features
from fusion import FatigueState, LEVEL_INFO


def draw_bar(overlay, bx, by, bw, bh, value, max_val, fill_col, label, thresholds=()):
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
    fill = int(bw * min(value / max_val, 1.0))
    if fill > 0:
        cv2.rectangle(overlay, (bx, by), (bx + fill, by + bh), fill_col, -1)
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (80, 80, 80), 1)
    for pct, tcol in thresholds:
        mx = bx + int(bw * pct)
        cv2.line(overlay, (mx, by - 2), (mx, by + bh + 2), tcol, 1)
    cv2.putText(overlay, label, (bx + 4, by + bh - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)


def render(frame, feats, fusion_result):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    face_found  = feats["face_found"]
    lbox, rbox, mbox = feats["lbox"], feats["rbox"], feats["mbox"]
    pitch       = feats["pitch"]
    yaw         = feats["yaw"]

    perclos     = fusion_result["perclos"]
    eye_streak  = fusion_result["eye_streak"]
    smooth_yawn = fusion_result["smooth_yawn"]
    yawn_drowsy = fusion_result["yawn_drowsy"]
    pitch_abs   = fusion_result["pitch_abs"]
    ema_score   = fusion_result["ema_score"]
    level       = fusion_result["level"]

    state_text, state_color = LEVEL_INFO[level]

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

    # Tam genişlik skor çubuğu
    bar_y, bar_h = 55, 12
    cv2.rectangle(overlay, (0, bar_y), (w, bar_y + bar_h), (30, 30, 30), -1)
    bar_fill = int(w * ema_score / 100.0)
    if bar_fill > 0:
        cv2.rectangle(overlay, (0, bar_y), (bar_fill, bar_y + bar_h), state_color, -1)
    for pct, col in [(SCORE_WARN / 100, (0, 200, 200)), (SCORE_DROWS / 100, (0, 80, 255))]:
        mx = int(w * pct)
        cv2.line(overlay, (mx, bar_y), (mx, bar_y + bar_h), col, 1)

    # Sol panel
    px, py   = 10, 73
    pw_panel = 390
    ph_panel = 240
    cv2.rectangle(overlay, (px, py), (px + pw_panel, py + ph_panel), (18, 18, 18), -1)
    cv2.rectangle(overlay, (px, py), (px + pw_panel, py + ph_panel), (60, 60, 60),  1)

    def lbl(text, vy, color=(210, 210, 210)):
        cv2.putText(overlay, text, (px + 12, vy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1)

    eye_c    = (0, 80, 255) if perclos > 0.25          else (0, 200, 80)
    yawn_c   = (0, 80, 255) if yawn_drowsy              else (0, 200, 80)
    streak_c = (0, 80, 255) if eye_streak               else (0, 200, 80)
    pitch_c  = (0, 80, 255) if pitch_abs > HEAD_PITCH_WARN else (0, 200, 80)

    lbl(f"Skor    : {ema_score:5.1f} / 100", py + 28,  state_color)
    lbl(f"PERCLOS : {perclos * 100:5.1f}%",  py + 53,  eye_c)
    lbl(f"Streak  : {'EVET (' + str(EYE_STREAK_FRAMES) + ' frame)' if eye_streak else 'Yok'}",
        py + 78, streak_c)
    lbl(f"Bas Pitch: {pitch:+5.1f}  Yaw: {yaw:+5.1f} deg", py + 103, pitch_c)
    lbl(f"Esname  : {smooth_yawn * 100:5.1f}%  {'ESNEME' if yawn_drowsy else 'NORMAL'}",
        py + 128, yawn_c)

    cv2.line(overlay, (px + 10, py + 143), (px + pw_panel - 10, py + 143), (60, 60, 60), 1)

    bx0 = px + 10
    bw0 = pw_panel - 20
    bh0 = 16

    pcol = (0, 40, 220) if perclos > 0.40 else ((0, 160, 220) if perclos > 0.25 else (0, 180, 80))
    draw_bar(overlay, bx0, py + 153, bw0, bh0, perclos, 1.0, pcol,
             "PERCLOS  |25%      |40%",
             [(0.25, (200, 200, 0)), (0.40, (0, 100, 255))])

    ycol = (0, 40, 220) if yawn_drowsy else (0, 180, 80)
    draw_bar(overlay, bx0, py + 181, bw0, bh0, smooth_yawn, 1.0, ycol,
             f"Esname olasiligi  |{int(YAWN_THRESH * 100)}%",
             [(YAWN_THRESH, (200, 200, 0))])

    pitch_norm = min(abs(pitch) / 40.0, 1.0)
    pcol2 = (0, 40, 220) if pitch_abs > HEAD_PITCH_ALERT else (
             (0, 165, 255) if pitch_abs > HEAD_PITCH_WARN else (0, 180, 80))
    draw_bar(overlay, bx0, py + 209, bw0, bh0, pitch_norm, 1.0, pcol2,
             f"Bas pitch  |{int(HEAD_PITCH_WARN)}d   |{int(HEAD_PITCH_ALERT)}d",
             [(HEAD_PITCH_WARN / 40, (200, 200, 0)),
              (HEAD_PITCH_ALERT / 40, (0, 100, 255))])

    if not face_found:
        cv2.putText(overlay, "YUZ BULUNAMADI", (w // 2 - 130, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 2)

    return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = FatigueState()
    print("Kamera acildi. Cikis icin 'q' tusuna basin.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        feats = extract_features(frame)
        fusion_result = state.update(
            feats["eye_prob"], feats["yawn_prob"], feats["pitch"]
        )

        out = render(frame, feats, fusion_result)
        cv2.imshow("Drowsiness Fusion  |  q = cikis", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
