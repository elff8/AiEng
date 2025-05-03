"""
Train pronunciation models.

Usage
-----
python -m pronunciation_scoring.train --word apple --epochs 80
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from . import config, io, model as model_factory

# -----------------------------------------------------------------------------  
def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Train per-word pronunciation model")
    p.add_argument("--word", required=True,
                   help="The word whose recordings exist under DatasetAudio/<word>")
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--epochs",   type=int, default=config.EPOCHS)
    p.add_argument("--batch",    type=int, default=config.BATCH_SIZE)
    args = p.parse_args(argv)

    word_dir = Path(args.data_dir) / args.word
    if not word_dir.exists():
        raise SystemExit(f"❌ dataset folder not found: {word_dir}")

    print(f"📂  Loading dataset from {word_dir} …")
    X, y = io.load_dataset_from_scored_subdirs(word_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, scaler = io.fit_transform_scaler(X_train)
    X_val   = np.array([scaler.transform(x) for x in X_val])

    net = model_factory.create_model()
    early = EarlyStopping(monitor="val_loss",
                          patience=5,
                          restore_best_weights=True)

    print("🚀  Training …")
    net.fit(X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch,
            validation_data=(X_val, y_val),
            verbose=2)

    # Evaluate once on the hold-out val set
    _, mae = net.evaluate(X_val, y_val, verbose=0)
    print(f"✅  Val MAE for “{args.word}” : {mae:.3f}")

    io.save_artifacts(args.word, net, scaler)


if __name__ == "__main__":
    main()
