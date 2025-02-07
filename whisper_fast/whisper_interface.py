import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
from typing import List

app = FastAPI()

# Whisper 모델 로드
model = whisper.load_model("base")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")
    
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")
    
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # 파일 저장
    with open(filepath, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Whisper를 사용하여 음성을 텍스트로 변환
    result = model.transcribe(filepath)
    
    # 파일 삭제
    os.remove(filepath)
    
    return JSONResponse(content={"transcription": result["text"]})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)
