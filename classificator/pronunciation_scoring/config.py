from pathlib import Path

SAMPLE_RATE = 16_000        # Hz
MAX_LEN     = 100           # MFCC frames per utterance
N_MFCC      = 13            # MFCC coefficients
BATCH_SIZE  = 8
EPOCHS      = 100

DATA_DIR   = Path("DatasetAudio")
MODELS_DIR = Path("models")         # artefacts saved here (<word>.keras + <word>.scaler)

MODELS_DIR.mkdir(exist_ok=True)     # auto-create on import
