# Face Recognition and Analysis System

This project provides a comprehensive real-time face recognition and facial analysis system using Python, OpenCV, Dlib, DeepFace, and the `face_recognition` library. The system detects faces, recognizes known individuals, and analyzes various facial attributes such as age, gender, emotions, and facial landmarks.

## Features
- **Face Detection**: Detects faces in real-time using Dlib and `face_recognition`.
- **Face Recognition**: Identifies known faces by comparing them with pre-encoded face data.
- **Facial Attribute Detection**: Extracts attributes such as age, gender, emotion, and race using DeepFace.
- **Facial Landmark Detection**: Detects 68 facial landmarks using Dlib's shape predictor.
- **Real-time Video Feed**: Captures live video feed from a webcam and processes frames to display recognition and analysis results.
- **Batch Image Processing**: Processes images to detect, recognize, and analyze faces in static image files.


## Example Usage
![alt text](images/individuals/Abdallah_Mohamed.jpeg)
#### Processed Abdallah_Mohamed.jpeg: {'age': 23, 'gender': {'Woman': 0.00031815122838452226, 'Man': 99.99967813491821}, 'dominant_emotion': 'happy', 'dominant_race': 'middle eastern'}
![alt text](<images/artifacts/Screenshot 2024-09-26 234743.png>)
#### Elon Musk: Age: 34 Dominant Emotion: Happy (97.22%) Dominant Gender: Man (99.73%) Dominant Race: Asian (33.31%) Confidence: 0.92 ...etc
![alt text](<images/artifacts/that is me.png>)

## Technologies
- **OpenCV**: Used for capturing video feed and image manipulation.
- **Dlib**: Used for detecting faces and predicting facial landmarks.
- **DeepFace**: Used for analyzing facial attributes such as age, gender, emotion, and race.
- **Face Recognition Library**: Used for encoding, comparing, and recognizing faces.
  
## Prerequisites
To run the project, you will need to install the following libraries:

```bash
pip install opencv-python dlib face_recognition deepface
```

Ensure you have the `shape_predictor_68_face_landmarks.dat` file, which can be downloaded from [Dlib's model page](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).


### How to Run the Project

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Abdallahelraey/Face-Recognition-and-Analysis-System.git
   ```

   ```bash  
   cd "Face Recognition"
   ```

2. **Prepare known faces**:  
   Place images of known individuals into the `./images/individuals` directory. The system will automatically encode and store their facial data for later recognition.

3. **Run the system**:  
   You can run the project in either **static** or **real-time video** mode by executing the `main.py` script. The system will prompt you to choose between these two modes.

   - **Run the face recognition system**:
     ```bash
     python main.py
     ```

   - The system will ask you to choose between:
     - `static`: To process a batch image.
     - `video`: To start real-time video recognition and analysis.

4. **Real-time video face recognition and analysis**:  
   If you choose `video`, the system will start capturing video using your webcam. It will recognize and display known faces and their locations in real time.
   - Press `q` to quit the video feed.

5. **Batch image face recognition and analysis**:  
   If you choose `static`, the system will load an image from the `./images/group/` directory (or any specified path) and perform face recognition and attribute analysis. The recognized faces, their locations, and their attributes will be printed in the console.

6. **Run other modules directly**:
   You can also run the two main face recognition modules directly without using `main.py`:
   - **Real-time video recognition**:
     ```bash
     python real_time_face_recognition_attributes_detention.py
     ```
   - **Static image recognition**:
     ```bash
     python static_face_recognition_attributes_detention.py
     ```

### Project Structure
- `main.py`: The main entry point to choose between static and video face recognition.
- `real_time_face_recognition_attributes_detention.py`: Handles real-time video-based face recognition.
- `static_face_recognition_attributes_detention.py`: Handles batch image face recognition and analysis.
- `./images/individuals/`: Directory to store images of known individuals for face recognition.
- `./images/group/`: Directory to store group images or test images for batch recognition.

--- 


## Customization
You can modify the paths for images, tune the facial analysis settings in the `DeepFace.analyze()` method, or extend the system to include additional features like face tracking.

## Example Usage

### Real-Time Video Feed
The system will:
1. Detect and recognize faces from the live camera feed.
2. Display information about each face, such as name, age, gender, and emotions.
3. Annotate the video feed with detected facial landmarks.

#### Processed Abdallah_Mohamed.jpeg: {'age': 23, 'gender': {'Woman': 0.00031815122838452226, 'Man': 99.99967813491821}, 'dominant_emotion': 'happy', 'dominant_race': 'middle eastern'}


## Contributing
If you wish to contribute to this project, please submit a pull request or open an issue.

## License
This project is licensed under the Apache License.

