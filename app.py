from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from static_face_recognition_attributes_detention import FaceRecognitionAndAnalysis
import shutil
import os
import uuid

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global instance (for demo; in production, use a better state management)
face_recognition_analysis = FaceRecognitionAndAnalysis()

@app.post("/load-known-faces/")
def load_known_faces(directory_or_image_path: str = Form(...)):
    """
    Load known faces from a directory or a single image path.
    """
    if not os.path.exists(directory_or_image_path):
        raise HTTPException(status_code=400, detail="Path does not exist.")
    face_recognition_analysis.load_known_faces(directory_or_image_path)
    return {"status": "success", "message": f"Loaded faces from {directory_or_image_path}"}

@app.post("/recognize/")
def recognize_and_analyze(image: UploadFile = File(...)):
    """
    Recognize and analyze faces in an uploaded image.
    """
    # Save uploaded file to a temp location
    temp_filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    try:
        results = face_recognition_analysis.recognize_and_analyze(temp_filename)
    except Exception as e:
        os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=str(e))
    os.remove(temp_filename)
    
    # Safely extract age if available
    if results and len(results) > 0 and 'attributes' in results[0] and 'age' in results[0]['attributes']:
        age = results[0]['attributes']['age']
        return JSONResponse(content={"results": age})
    else:
        return JSONResponse(content={"results": results, "message": "No age data available"})

@app.get("/")
def root():
    return {"message": "Face Recognition and Analysis API. Use /load-known-faces/ and /recognize/ endpoints."} 