import sys
import os

# Proje kökü — config.py'deki relative yollar (models/, face_landmarker.task) buraya göre
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # relative yolların çözümlenmesi için

import asyncio
import base64
import json
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import cv2
import mediapipe as mp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from detector import (
    face_landmarker,
    crop_eye_region, crop_mouth, get_head_pose,
)
from models import infer_eye, infer_yawn
from fusion import FatigueState
from config import LEFT_EYE, RIGHT_EYE

RTSP_URL = os.environ.get("RTSP_URL", "rtsp://mediamtx:8554/camera")
RTSP_INFER_FPS = 10  # saniyede kaç frame inference yapılsın


# ── Lifespan: RTSP reader thread'i başlat/durdur ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    stop_event = threading.Event()
    threading.Thread(target=_rtsp_grabber,        args=(stop_event,),       daemon=True).start()
    threading.Thread(target=_rtsp_inference_loop, args=(loop, stop_event),  daemon=True).start()
    print(f"RTSP reader başlatıldı: {RTSP_URL}")
    try:
        yield
    finally:
        stop_event.set()


app = FastAPI(lifespan=lifespan)


_latest_frame: tuple[float, np.ndarray] | None = None
_latest_lock = threading.Lock()


def _rtsp_grabber(stop_event: threading.Event):
    """
    Arka thread: RTSP'ten frame'leri olabildiğince hızlı okur,
    sadece EN SON frame'i belleğe yazar. Böylece buffer dolup
    geriden gelen kareleri işlemeyiz.
    """
    global _latest_frame

    while not stop_event.is_set():
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"RTSP açılamadı, 3sn sonra tekrar: {RTSP_URL}")
            time.sleep(3)
            continue

        print("RTSP grabber bağlandı.")
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("RTSP frame okunamadı, yeniden bağlanılıyor.")
                break
            with _latest_lock:
                _latest_frame = (time.time(), frame)

        cap.release()


def _rtsp_inference_loop(loop: asyncio.AbstractEventLoop, stop_event: threading.Event):
    """
    Inference thread: en son frame'i alır, modeli çalıştırır,
    sonucu broadcast eder. Inference süresi ne olursa olsun
    her zaman taze frame işlenir.
    """
    interval = 1.0 / RTSP_INFER_FPS
    last_ts = 0.0
    _pi_state.reset()

    while not stop_event.is_set():
        time.sleep(interval)

        with _latest_lock:
            snapshot = _latest_frame

        if snapshot is None:
            continue

        ts, frame = snapshot
        if ts == last_ts:
            continue   # aynı frame'i tekrar işleme
        last_ts = ts

        response = process_frame(frame, _pi_state)
        asyncio.run_coroutine_threadsafe(_broadcast(json.dumps(response)), loop)


def decode_frame(data_url: str) -> np.ndarray | None:
    """Base64 data URL'yi BGR numpy frame'e çevirir."""
    try:
        _, encoded = data_url.split(",", 1)
        buf = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def process_frame(frame: np.ndarray, state: FatigueState) -> dict:
    """
    Tek frame'den yorgunluk tahmini ve landmark listesi üretir.

    MediaPipe yalnızca bir kez çalışır; sonuç hem feature extraction
    hem de landmark listesi için kullanılır.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = face_landmarker.detect(mp_image)

    if not detection.face_landmarks:
        return {
            "level": 0, "ema_score": 0.0,
            "perclos": 0.0, "eye_streak": False,
            "smooth_yawn": 0.0, "yawn_drowsy": False,
            "pitch": 0.0, "yaw": 0.0,
            "face_found": False, "landmarks": [],
        }

    lm = detection.face_landmarks[0]

    # Göz kapanma olasılığı
    left_crop,  _ = crop_eye_region(frame, lm, LEFT_EYE,  w, h)
    right_crop, _ = crop_eye_region(frame, lm, RIGHT_EYE, w, h)
    eye_prob = (infer_eye(left_crop) + infer_eye(right_crop)) / 2.0

    # Esneme olasılığı
    mouth_crop, _ = crop_mouth(frame, lm, w, h)
    yawn_prob = infer_yawn(mouth_crop)

    # Baş açısı
    pitch, yaw = get_head_pose(lm, w, h)

    # Yorgunluk skoru
    result = state.update(eye_prob, yawn_prob, pitch)

    # Landmark listesi (frontend formatı: x,y,z normalize koordinatlar)
    landmarks = [{"x": p.x, "y": p.y, "z": p.z} for p in lm]

    return {
        "level":       result["level"],
        "ema_score":   round(result["ema_score"], 1),
        "perclos":     round(result["perclos"] * 100, 1),
        "eye_streak":  result["eye_streak"],
        "smooth_yawn": round(result["smooth_yawn"] * 100, 1),
        "yawn_drowsy": result["yawn_drowsy"],
        "pitch":       round(pitch, 1),
        "yaw":         round(yaw, 1),
        "face_found":  True,
        "landmarks":   landmarks,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Webcam test endpoint — frontend webcam'den frame alır, tahmin döndürür."""
    await websocket.accept()
    state = FatigueState()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") != "frame":
                continue

            frame = decode_frame(msg["data"])
            if frame is None:
                continue

            response = process_frame(frame, state)
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        pass


# ── MediaMTX RTSP yayını → control sayfası köprüsü ───────────────────────────
# Backend, Pi'nin yayınını MediaMTX'ten RTSP olarak çeker, modeli çalıştırır,
# sonucu /ws/control dinleyicilerine broadcast eder.

_pi_state: FatigueState = FatigueState()
_control_clients: set[WebSocket] = set()


async def _broadcast(message: str) -> None:
    """Tüm control dinleyicilerine mesaj gönder; kopuk olanları temizle."""
    dead = []
    for client in _control_clients:
        try:
            await client.send_text(message)
        except Exception:
            dead.append(client)
    for d in dead:
        _control_clients.discard(d)


@app.websocket("/ws/control")
async def control_subscriber(websocket: WebSocket):
    """Frontend control sayfası tahmin sonuçlarını dinler."""
    await websocket.accept()
    _control_clients.add(websocket)

    try:
        while True:
            await websocket.receive_text()  # heartbeat / no-op
    except WebSocketDisconnect:
        pass
    finally:
        _control_clients.discard(websocket)
