import os
import face_recognition
from deepface import DeepFace
import numpy as np
import cv2
import dlib

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
        
        result = {}
        
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
            
            result = attributes['age']
        
        return result

    def detect_eye_redness(self, face_image):
        # Convert to BGR if needed
        if len(face_image.shape) == 2 or face_image.shape[2] == 1:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
        elif face_image.shape[2] == 4:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2BGR)
        # Use Haar cascade for eye detection
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
        redness_scores = []
        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_image[ey:ey+eh, ex:ex+ew]
            r = np.mean(eye_roi[:,:,2])
            g = np.mean(eye_roi[:,:,1])
            b = np.mean(eye_roi[:,:,0])
            redness = r - (g + b) / 2
            redness_scores.append(redness)
        return float(np.mean(redness_scores)) if redness_scores else None

    def analyze_under_eye_bags(self, face_image, predictor_path='shape_predictor_68_face_landmarks.dat'):
        import os
        try:
            if not os.path.exists(predictor_path):
                print(f"Predictor file not found: {predictor_path}")
                return {
                    "left_eye_bag_darkness": None,
                    "right_eye_bag_darkness": None,
                    "deep_under_eye_bags": False,
                    "possible_health_conditions": ["predictor file not found"]
                }
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)
            rects = detector(gray, 1)
            if not rects:
                print("No face detected in under-eye analysis region.")
                return {
                    "left_eye_bag_darkness": None,
                    "right_eye_bag_darkness": None,
                    "deep_under_eye_bags": False,
                    "possible_health_conditions": ["no face detected"]
                }
            shape = predictor(gray, rects[0])
            left_eye_bottom = shape.part(39)
            right_eye_bottom = shape.part(45)
            left_cheek = shape.part(31)
            right_cheek = shape.part(35)
            h, w = gray.shape
            def safe_slice(y1, y2, x1, x2):
                return gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            # Under-eye regions
            left_region = safe_slice(left_eye_bottom.y, left_cheek.y, left_eye_bottom.x-10, left_eye_bottom.x+10)
            right_region = safe_slice(right_eye_bottom.y, right_cheek.y, right_eye_bottom.x-10, right_eye_bottom.x+10)
            # Cheek reference regions (just below cheek landmarks)
            left_cheek_y1 = left_cheek.y + 10
            left_cheek_y2 = left_cheek_y1 + 20
            left_cheek_x1 = left_cheek.x - 10
            left_cheek_x2 = left_cheek.x + 10
            left_cheek_region = safe_slice(left_cheek_y1, left_cheek_y2, left_cheek_x1, left_cheek_x2)
            right_cheek_y1 = right_cheek.y + 10
            right_cheek_y2 = right_cheek_y1 + 20
            right_cheek_x1 = right_cheek.x - 10
            right_cheek_x2 = right_cheek.x + 10
            right_cheek_region = safe_slice(right_cheek_y1, right_cheek_y2, right_cheek_x1, right_cheek_x2)
            # Means
            left_darkness = np.mean(left_region) if left_region.size > 0 else None
            right_darkness = np.mean(right_region) if right_region.size > 0 else None
            left_cheek_mean = np.mean(left_cheek_region) if left_cheek_region.size > 0 else None
            right_cheek_mean = np.mean(right_cheek_region) if right_cheek_region.size > 0 else None
            # Relative comparison
            left_relative = left_darkness - left_cheek_mean if (left_darkness is not None and left_cheek_mean is not None) else None
            right_relative = right_darkness - right_cheek_mean if (right_darkness is not None and right_cheek_mean is not None) else None
            # Threshold: flag as bag if under-eye is at least 15 units darker than cheek
            rel_threshold = -15
            left_bag = left_relative is not None and left_relative < rel_threshold
            right_bag = right_relative is not None and right_relative < rel_threshold
            possible_conditions = []
            if left_bag or right_bag:
                possible_conditions = ["chronic fatigue", "poor sleep", "adrenal stress"]
            return {
                "left_eye_bag_darkness": left_darkness,
                "right_eye_bag_darkness": right_darkness,
                "left_relative_darkness": left_relative,
                "right_relative_darkness": right_relative,
                "deep_under_eye_bags": left_bag or right_bag,
                "possible_health_conditions": possible_conditions
            }
        except Exception as e:
            print(f"Error in under-eye bag analysis: {str(e)}")
            return {
                "left_eye_bag_darkness": None,
                "right_eye_bag_darkness": None,
                "left_relative_darkness": None,
                "right_relative_darkness": None,
                "deep_under_eye_bags": False,
                "possible_health_conditions": [f"error: {str(e)}"]
            }

    def _analyze_unknown_face(self, unknown_image, top, bottom, left, right):
        try:
            face_image = unknown_image[top:bottom, left:right]
            attributes = DeepFace.analyze(img_path=face_image, 
                                          actions=['age', 'gender', 'emotion', 'race'],
                                          enforce_detection=False)
            if isinstance(attributes, list):
                attributes = attributes[0]
            eye_redness = self.detect_eye_redness(face_image)
            under_eye_bags = self.analyze_under_eye_bags(face_image)
            attributes = {
                'age': attributes['age'],
                'gender': attributes['gender'],
                'dominant_emotion': attributes['dominant_emotion'],
                'dominant_race': attributes['dominant_race'],
                'eye_redness': eye_redness,
                'under_eye_bags': under_eye_bags
            }
        except Exception as e:
            print(f"Error analyzing face: {str(e)}")
            attributes = None
        print(f"attributes: ", attributes)
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