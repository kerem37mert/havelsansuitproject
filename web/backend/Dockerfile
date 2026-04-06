FROM python:3.11-slim

# OpenCV ve MediaPipe için sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Önce bağımlılıkları kur (layer cache için)
COPY web/backend/requirements.txt web/backend/requirements.txt
RUN pip install --no-cache-dir \
    opencv-python-headless \
    -r web/backend/requirements.txt

# Proje dosyalarını kopyala
COPY config.py fusion.py detector.py models.py ./
COPY web/backend/main.py web/backend/main.py
COPY models/ models/
COPY face_landmarker.task .

CMD ["python", "-m", "uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
