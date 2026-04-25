"""
Raspberry Pi → Backend frame yayıncısı.

picamera2 ile yakalanan frame'leri JPEG'e encode edip
WebSocket üzerinden backend'in /ws/pi endpoint'ine gönderir.
Backend modeli çalıştırır, sonucu /ws/control dinleyicilerine yayınlar.
"""

import asyncio
import base64
import json
import signal

import cv2
import websockets
from picamera2 import Picamera2

WS_URL  = "wss://api.havelsansuitproject.dev/ws/pi"
WIDTH   = 640
HEIGHT  = 480
FPS     = 10        # backend inference saniyede ~10 frame işliyor, yeterli
JPEG_Q  = 70


async def stream():
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"},
        controls={"FrameRate": FPS},
    ))
    cam.start()

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q]
    interval = 1.0 / FPS

    async with websockets.connect(WS_URL, max_size=4 * 1024 * 1024) as ws:
        print(f"Bağlandı: {WS_URL}")

        stop = asyncio.Event()
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        while not stop.is_set():
            arr = cam.capture_array()
            ok, buf = cv2.imencode(".jpg", arr, encode_params)
            if not ok:
                continue

            data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
            await ws.send(json.dumps({"type": "frame", "data": data_url}))
            await asyncio.sleep(interval)

    cam.stop()
    print("Durduruldu.")


if __name__ == "__main__":
    asyncio.run(stream())
