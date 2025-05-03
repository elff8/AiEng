from __future__ import annotations
import numpy as np
import librosa

from . import config

# -----------------------------------------------------------------------------  
def augment(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Simple augmentation pipeline: time-stretch → pitch-shift → add white noise.
    """
    # 1. Time-stretch (±10 %)
    y_aug = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
    # 2. Pitch-shift (±1 semitone)
    y_aug = librosa.effects.pitch_shift(y=y_aug, sr=sr,
                                        n_steps=np.random.uniform(-1, 1))
    # 3. Add white noise (0.3 % amplitude)
    noise = 0.003 * np.random.randn(len(y_aug))
    return y_aug + noise


def extract_features(file_path: str | bytes | "os.PathLike",
                     *,
                     do_aug: bool = False) -> np.ndarray:
    """
    Load audio → (optionally) augment → compute MFCC → pad / truncate.

    Returns
    -------
    np.ndarray shape (config.MAX_LEN, config.N_MFCC)
    """
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)

    if do_aug:
        y = augment(y, sr)

    mfcc = librosa.feature.mfcc(y=y,
                                sr=sr,
                                n_mfcc=config.N_MFCC)          # (n_mfcc, time)

    # Pad / truncate along the *time* axis
    if mfcc.shape[1] < config.MAX_LEN:
        pad = config.MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
    else:
        mfcc = mfcc[:, :config.MAX_LEN]

    return mfcc.T       # (time, n_mfcc)  ==  (MAX_LEN, N_MFCC)
