import os
import face_recognition
from deepface import DeepFace
import numpy as np

class FaceRecognitionAndAnalysis:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_attributes = []

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

    def recognize_and_analyze(self, image_path):
        unknown_image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        
        results = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            attributes = None
            
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                attributes = self.known_face_attributes[first_match_index]
            
            if not attributes:
                attributes = self._analyze_unknown_face(unknown_image, top, bottom, left, right)
            
            results.append({
                'name': name,
                'location': (top, right, bottom, left),
                'attributes': attributes
            })
        
        return results

    def _analyze_unknown_face(self, unknown_image, top, bottom, left, right):
        try:
            face_image = unknown_image[top:bottom, left:right]
            attributes = DeepFace.analyze(img_path=face_image, 
                                          actions=['age', 'gender', 'emotion', 'race'],
                                          enforce_detection=False)
            
            if isinstance(attributes, list):
                attributes = attributes[0]
            
            attributes = {
                'age': attributes['age'],
                'gender': attributes['gender'],
                'dominant_emotion': attributes['dominant_emotion'],
                'dominant_race': attributes['dominant_race']
            }
        except Exception as e:
            print(f"Error analyzing face: {str(e)}")
            attributes = None
        
        return attributes

# Example usage
if __name__ == "__main__":
    face_recognition_analysis = FaceRecognitionAndAnalysis()
    # Load known faces from a directory
    face_recognition_analysis.load_known_faces("./images/individuals")
    # Load known faces from a single image
    # face_recognition_analysis.load_known_faces(".\images\individuals\single_image.png")
    test_image_path = ".\images\group\group_image.png"
    results = face_recognition_analysis.recognize_and_analyze(test_image_path)
    
    for result in results:
        print(f"Name: {result['name']}")
        print(f"Location: {result['location']}")
        print(f"Attributes: {result['attributes']}")
        print("---")