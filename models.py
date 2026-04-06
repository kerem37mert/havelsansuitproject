import numpy as np
import cv2
import tensorflow as tf
from config import EYE_MODEL_PATH, YAWN_MODEL_PATH, EYE_IMG_SIZE, YAWN_IMG_SIZE


def _load_models():
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    # Eski modellerdeki bilinmeyen config anahtarını yoksay
    _orig = tf.keras.layers.Dense.from_config.__func__
    @classmethod
    def _patched(cls, config):
        config.pop('quantization_config', None)
        return _orig(cls, config)
    tf.keras.layers.Dense.from_config = _patched

    print("Modeller yukleniyor...")
    eye  = tf.keras.models.load_model(EYE_MODEL_PATH,  compile=False)
    yawn = tf.keras.models.load_model(YAWN_MODEL_PATH, compile=False)
    print("Modeller hazir.")
    return eye, yawn


eye_model, yawn_model = _load_models()


def infer_eye(crop: np.ndarray) -> float:
    """BGR göz kırpığından kapanma olasılığı döndürür [0-1]."""
    if crop.size == 0:
        return 0.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (EYE_IMG_SIZE, EYE_IMG_SIZE)).astype(np.float32) / 255.0
    raw = eye_model(np.expand_dims(img, 0), training=False)
    return float(tf.cast(raw, tf.float32).numpy().flatten()[0])


def infer_yawn(crop: np.ndarray) -> float:
    """BGR ağız kırpığından esneme olasılığı döndürür [0-1]."""
    if crop.size == 0:
        return 0.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (YAWN_IMG_SIZE, YAWN_IMG_SIZE)).astype(np.float32) / 255.0
    raw = yawn_model(np.expand_dims(img, 0), training=False)
    return float(tf.cast(raw, tf.float32).numpy().flatten()[0])
