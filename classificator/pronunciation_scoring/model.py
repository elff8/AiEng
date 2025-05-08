import tensorflow as tf
from . import config

# -----------------------------------------------------------------------------  
def create_model(max_len: int = config.MAX_LEN,
                 n_mfcc: int = config.N_MFCC) -> tf.keras.Model:
    """
    Conv1D-based regressor with global average pooling and two FC layers.
    """
    inputs = tf.keras.layers.Input(shape=(max_len, n_mfcc))

    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)   # ∈ [0, 1]

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )
    return model
