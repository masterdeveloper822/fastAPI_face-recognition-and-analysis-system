from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from static_face_recognition_attributes_detention import FaceRecognitionAndAnalysis
import shutil
import os
import uuid
import traceback

app = FastAPI()

# Global instance (for demo; in production, use a better state management)
face_recognition_analysis = FaceRecognitionAndAnalysis()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/load-known-faces/")
def load_known_faces(directory_or_image_path: str = Form(...)):
    if not os.path.exists(directory_or_image_path):
        raise HTTPException(status_code=400, detail="Path does not exist.")
    face_recognition_analysis.load_known_faces(directory_or_image_path)
    return {"status": "success", "message": f"Loaded faces from {directory_or_image_path}"}

@app.post("/recognize/")
def recognize_and_analyze(image: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    try:
        results = face_recognition_analysis.recognize_and_analyze(temp_filename)
    except Exception as e:
        os.remove(temp_filename)
        tb = traceback.format_exc()
        print("Exception in /recognize/:\n", tb)
        raise HTTPException(status_code=500, detail=tb)
    os.remove(temp_filename)
    return JSONResponse(content={"results": results})

@app.get("/")
def root():
    return {"message": "Face Recognition and Analysis API. Use /load-known-faces/ and /recognize/ endpoints."} 