import argparse, pathlib
import joblib
import tensorflow as tf
from pronunciation_scoring import features, io


p = argparse.ArgumentParser()
p.add_argument("--word", required=True)
p.add_argument("--audio", required=True)
p.add_argument("--model-dir", default="models")
args = p.parse_args()

model_path  = pathlib.Path(args.model_dir) / f"{args.word}.keras"
scaler_path = pathlib.Path(args.model_dir) / f"{args.word}.scaler"

model  = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

x = features.extract_features(args.audio)
x = scaler.transform(x)[None, ...]          # shape (1, MAX_LEN, N_MFCC)
score = float(model.predict(x, verbose=0))
print(f"Score: {score*10:.1f}/10")
