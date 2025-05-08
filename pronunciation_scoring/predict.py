import argparse, pathlib
import joblib
import tensorflow as tf
from pronunciation_scoring import features, io


def predict_score(word: str, audio_path: str, model_dir: str = "models") -> float:
    model_path = pathlib.Path(model_dir) / f"{word}.keras"
    scaler_path = pathlib.Path(model_dir) / f"{word}.scaler"

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    x = features.extract_features(audio_path)
    x = scaler.transform(x)[None, ...]  # shape (1, MAX_LEN, N_MFCC)
    score = float(model.predict(x, verbose=0))

    return round(score * 10, 1)
