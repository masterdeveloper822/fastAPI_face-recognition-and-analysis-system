# Face Recognition and Analysis System (Web Backend)

This project now provides a web backend (API) for static image face recognition and facial attribute analysis using FastAPI.

## Features
- **Face Detection & Recognition**: Detects and recognizes faces in static images.
- **Facial Attribute Detection**: Extracts age, gender, emotion, and race using DeepFace.
- **REST API**: Interact with the system via HTTP endpoints.

## How to Run the Web Backend

1. **Install dependencies** (if not already):
   ```bash
   pip install -r requirements.txt
   ```

   python version: 

2. **Prepare known faces**:
   Place images of known individuals into the `./images/individuals` directory. The system will encode and store their facial data for recognition.

3. **Start the FastAPI server**:
   ```bash
   uvicorn app:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000/`.

## API Usage

### 1. Load Known Faces
POST `/load-known-faces/`
- **Form field:** `directory_or_image_path` (path to directory or image)
- **Example (using curl):**
  ```bash
  curl -X POST -F "directory_or_image_path=./images/individuals" http://127.0.0.1:8000/load-known-faces/
  ```

### 2. Recognize and Analyze Faces in an Image
POST `/recognize/`
- **Form field:** `image` (file upload)
- **Example (using curl):**
  ```bash
  curl -X POST -F "image=@/path/to/image.jpg" http://127.0.0.1:8000/recognize/
  ```
- **Response:** JSON with recognized faces, locations, and attributes.

## Notes
- **No video detection:** This backend only supports static image analysis.
- **Old CLI and video modes are deprecated.**
- Ensure you have the `shape_predictor_68_face_landmarks.dat` file in the project directory.

## License
See LICENSE for details.

