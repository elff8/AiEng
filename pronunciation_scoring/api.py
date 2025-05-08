from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import subprocess
import uuid
import os

app = FastAPI()

@app.post("/predict")
async def predict_audio(
    word: str = Form(...),
    file: UploadFile = File(...)
):
    ext = file.filename.split(".")[-1]
    temp_filename = f"temp_{uuid.uuid4()}.{ext}"
    temp_path = os.path.join("temp_files", temp_filename)
    os.makedirs("temp_files", exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        result = subprocess.run(
            ["python", "pronunciation_scoring/predict.py", "--word", word, "--audio", temp_path],
            capture_output=True, text=True
        )
    finally:
        os.remove(temp_path)

    if result.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={"error": "Model execution failed", "details": result.stderr}
        )

    return {"score": result.stdout.strip()}
