from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Iterable

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from . import config, features


# -----------------------------------------------------------------------------  
def load_dataset_from_scored_subdirs(word_dir: str | Path,
                                     *,
                                     extract_fn=features.extract_features,
                                     aug_prob: float = 0.5
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads directories 0.0, 0.4, 0.8 … under <word_dir> and returns X, y.
    """
    X, y = [], []
    word_dir = Path(word_dir)

    for score_dir in sorted(word_dir.iterdir()):
        if not score_dir.is_dir():
            continue
        try:
            score = float(str(score_dir.name).replace(",", "."))
        except ValueError:
            print(f"⚠️  Skip non-numeric folder: {score_dir.name}")
            continue

        for wav in score_dir.glob("*.wav"):
            X.append(extract_fn(wav, do_aug=np.random.rand() < aug_prob))
            y.append(score)

        for mp3 in score_dir.glob("*.mp3"):
            X.append(extract_fn(mp3, do_aug=np.random.rand() < aug_prob))
            y.append(score)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y


# -----------------------------------------------------------------------------  
def fit_transform_scaler(X: np.ndarray,
                         scaler: StandardScaler | None = None
                         ) -> Tuple[np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler frame-wise and return transformed X.
    """
    if scaler is None:
        scaler = StandardScaler()

    frames = X.reshape(-1, X.shape[-1])          # (#frames, n_mfcc)
    scaler.fit(frames)

    X_scaled = np.array([scaler.transform(xx) for xx in X])
    return X_scaled, scaler


# -----------------------------------------------------------------------------  
def save_artifacts(word: str,
                   model,
                   scaler: StandardScaler,
                   model_dir: str | Path = config.MODELS_DIR):
    """
    Save model (.keras) and scaler (.scaler) side-by-side.
    """
    model_dir = Path(model_dir)
    model_path  = model_dir / f"{word}.keras"
    scaler_path = model_dir / f"{word}.scaler"

    model.save(model_path)                # Keras v3 native format
    joblib.dump(scaler, scaler_path)

    print(f"💾  Saved:\n • {model_path}\n • {scaler_path}")


def load_artifacts(word: str,
                   model_dir: str | Path = config.MODELS_DIR):
    """
    Load model + scaler pair for inference.
    """
    from tensorflow.keras.models import load_model

    model_dir = Path(model_dir)
    model  = load_model(model_dir / f"{word}.keras")
    scaler = joblib.load(model_dir / f"{word}.scaler")
    return model, scaler
