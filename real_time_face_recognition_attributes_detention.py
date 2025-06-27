import cv2
import dlib
from deepface import DeepFace
import face_recognition
import os
import time

class FaceRecognitionSystem:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.video_capture = cv2.VideoCapture(0)
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_attributes = []

    def convert_frame_to_gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def convert_frame_to_rgb(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def detect_faces_dlib(self, gray_frame):
        return self.detector(gray_frame)

    def detect_faces_face_recognition(self, rgb_frame):
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return face_locations, face_encodings

    def process_frame(self, frame, known_face_encodings, known_face_names):
        gray_frame = self.convert_frame_to_gray(frame)
        rgb_frame = self.convert_frame_to_rgb(frame)
        dlib_faces = self.detect_faces_dlib(gray_frame)
        face_locations, face_encodings = self.detect_faces_face_recognition(rgb_frame)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_image = frame[top:bottom, left:right]
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            try:
                analysis = DeepFace.analyze(face_image_rgb, enforce_detection=False)
                age = analysis[0]['age']
                gender = analysis[0]['dominant_gender']
                emotion = analysis[0]['dominant_emotion']

                cv2.putText(frame, f'Name: {name}', (left, top - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Age: {age}', (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Gender: {gender}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Emotion: {emotion}', (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except ValueError as e:
                print(f"Error analyzing face: {str(e)}")
                continue

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            dlib_face = dlib.rectangle(left, top, right, bottom)
            landmarks = self.predictor(gray_frame, dlib_face)
            for n in range(0, 68):  # Loop over the 68 facial landmarks
                x_point = landmarks.part(n).x
                y_point = landmarks.part(n).y
                cv2.circle(frame, (x_point, y_point), 2, (0, 255, 0), -1)

        return frame

    def display_video_feed(self, frame):
        cv2.imshow('Video', frame)

    def capture_video_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            print("Failed to grab frame")
            return None
        return frame

    def release_resources(self):
        self.video_capture.release()
        cv2.destroyAllWindows()

    def load_known_faces(self, directory_or_image_path):
        if os.path.isdir(directory_or_image_path):
            self._load_known_faces_from_directory(directory_or_image_path)
        else:
            self._load_known_faces_from_single_image(directory_or_image_path)

    def _load_known_faces_from_directory(self, directory_path):
        for filename in os.listdir(directory_path):
            if filename.endswith((".jpeg", ".jpg", ".png")):
                image_path = f"{directory_path}/{filename}"
                self._process_image(image_path)

    def _load_known_faces_from_single_image(self, image_path):
        self._process_image(image_path)

    def _process_image(self, image_path):
        known_image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(known_image)
        
        if face_locations:
            known_encoding = face_recognition.face_encodings(known_image, face_locations)[0]
            self.known_face_encodings.append(known_encoding)
            self.known_face_names.append(image_path.split("/")[-1].split(".")[0])
            self._detect_face_attributes(image_path)
        else:
            print(f"No face detected in {image_path}")

    def _detect_face_attributes(self, image_path):
        try:
            attributes = DeepFace.analyze(img_path=image_path, 
                                          actions=['age', 'gender', 'emotion', 'race'],
                                          enforce_detection=False)
            
            if isinstance(attributes, list):
                attributes = attributes[0]
            
            face_attributes = {
                'age': attributes['age'],
                'gender': attributes['gender'],
                'dominant_emotion': attributes['dominant_emotion'],
                'dominant_race': attributes['dominant_race']
            }
            self.known_face_attributes.append(face_attributes)
            
            print(f"Processed {image_path}: {face_attributes}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            self.known_face_attributes.append(None)

def main():
    face_recognition_system = FaceRecognitionSystem()
    face_recognition_system.load_known_faces("./images/individuals")  
    start_time = time.time()

    while time.time() - start_time < 10:
        frame = face_recognition_system.capture_video_frame()
        if frame is None:
            break

        frame = face_recognition_system.process_frame(frame, face_recognition_system.known_face_encodings, face_recognition_system.known_face_names)
        face_recognition_system.display_video_feed(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_recognition_system.release_resources()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()