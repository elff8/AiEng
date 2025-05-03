"""
Pronunciation scoring with Conv1D + global StandardScaler
---------------------------------------------------------
Сохраняет:
    • <word>_model.h5   – веса нейросети
    • <word>_scaler.pkl – обученный StandardScaler
"""

import os
import pickle
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------  ПАРАМЕТРЫ  -------------------------
SAMPLE_RATE = 16_000        # частота дискретизации
MAX_LEN     = 50            # число кадров MFCC на запись
N_MFCC      = 13            # коэффициентов MFCC
DATA_DIR    = "DatasetAudio"
BATCH_SIZE  = 8
EPOCHS      = 50

# --------------------  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ  ----------------
def extract_features(file_path: str) -> np.ndarray:
    """
    Загружает wav, считает MFCC и приводит длину к MAX_LEN кадров.
    Нормализация НЕ выполняется – этим займётся глобальный scaler.
    Возвращает ndarray формы (MAX_LEN, N_MFCC).
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)  # (n_mfcc, time)

    # pad / truncate до одинаковой длины
    if mfcc.shape[1] < MAX_LEN:
        pad = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc.T                     # (time, n_mfcc) = (MAX_LEN, N_MFCC)


def load_dataset_from_scored_subdirs(word_dir: str):
    """
    Читает подкаталоги 0.0, 0.4, 0.8 ...,
    собирает X (MFCC) и y (оценку от 0.0 до 1.0).
    """
    X, y = [], []
    for score_dir in os.listdir(word_dir):
        path = os.path.join(word_dir, score_dir)
        if not os.path.isdir(path):
            continue
        try:
            score = float(score_dir.replace(",", "."))   # на случай 0,4
        except ValueError:
            print(f"Пропущена папка: {score_dir} (не число)")
            continue

        for fname in os.listdir(path):
            if not fname.endswith(".wav"):
                continue
            fpath = os.path.join(path, fname)
            X.append(extract_features(fpath))
            y.append(score)

    return np.array(X), np.array(y, dtype=np.float32)


def create_model() -> tf.keras.Model:
    """
    Conv1D → BatchNorm → ReLU → MaxPool  (x2)
    → GlobalAveragePooling → Dense → сигмоида.
    """
    inputs = tf.keras.layers.Input(shape=(MAX_LEN, N_MFCC))

    # Первый сверточный блок
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Второй сверточный блок
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # Глобальное усреднение по временным кадрам
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Полносвязные слои
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    # Используем явные ссылки на потери и метрики
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    return model



# -------------------------  TRAIN & SAVE  ----------------------
def train_and_save_model(word: str):
    """
    • Загружает датасет `DatasetAudio/<word>/<score>/<wav>`
    • Обучает scaler и модель
    • Сохраняет <word>_model.h5 и <word>_scaler.pkl
    """
    word_dir = os.path.join(DATA_DIR, word)
    X, y = load_dataset_from_scored_subdirs(word_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------- обучаем единый StandardScaler ----------
    scaler = StandardScaler()
    frames_for_scaler = X_train.reshape(-1, N_MFCC)        # (#frames, n_mfcc)
    scaler.fit(frames_for_scaler)

    # применяем scaler к train/test
    X_train = np.array([scaler.transform(x) for x in X_train])
    X_test  = np.array([scaler.transform(x) for x in X_test])

    # ---------- модель ----------
    model = create_model()
    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(X_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_split=0.2,
              callbacks=[early],
              verbose=2)

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"MAE для '{word}': {mae:.3f}")

    # ---------- сохранение ----------
    model.save(f"{word}_model.h5")
    with open(f"{word}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"Сохранены: {word}_model.h5  и  {word}_scaler.pkl")


# ---------------------------  PREDICT  -------------------------
def predict_word(model_path: str, scaler_path: str, audio_file: str):
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    feats = extract_features(audio_file)        # (MAX_LEN, N_MFCC)
    feats = scaler.transform(feats)             # нормализация
    feats = np.expand_dims(feats, axis=0)       # (1, MAX_LEN, N_MFCC)

    pred = model.predict(feats, verbose=0)[0][0]  # ∈ [0, 1]
    score10 = round(pred * 10, 1)
    print(f"Произношение: {score10}/10")

    if score10 >= 8:
        print("👍 Хорошо")
    elif score10 >= 5:
        print("😐 Средне")
    else:
        print("👎 Плохо")


# ---------------------------  MAIN  ----------------------------
if __name__ == "__main__":
    words = ["apple", "blueberry"]    # <-- ваши слова

    for w in words:
        print(f"\n=== Обучение модели для «{w}» ===")
        train_and_save_model(w)

    # --- пример предсказания ---
    predict_word("blueberry_model.h5",
                 "blueberry_scaler.pkl",
                 r"C:\Users\wisp\PycharmProjects\AIEng\TestAudio\a2.wav")
