from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uuid, os
from pronunciation_scoring.predict import predict_score
import logging

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
        score = predict_score(word, temp_path)
    except Exception as e:
        logging.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": "Model execution failed", "details": str(e)}
        )
    finally:
        os.remove(temp_path)

    return JSONResponse(
            status_code=200,
            content={"score": f"{score}/10"}
        )