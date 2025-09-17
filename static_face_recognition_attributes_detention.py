import os
import face_recognition
from deepface import DeepFace
import numpy as np
import cv2
import dlib

def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

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
            
            # Load the image to get face region for detailed analysis
            known_image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(known_image)
            
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_image = known_image[top:bottom, left:right]
                
                # Perform additional health analyses
                eye_redness = self.detect_eye_redness(face_image)
                under_eye_bags = self.analyze_under_eye_bags(face_image)
                droopy_eyelids = self.analyze_droopy_eyelids(face_image)
                forehead_wrinkles = self.analyze_forehead_wrinkles(face_image)
                acne_lesions = self.analyze_acne_lesions(face_image)
                
                face_attributes = convert_to_json_serializable({
                    'age': attributes['age'],
                    'gender': attributes['gender'],
                    'dominant_emotion': attributes['dominant_emotion'],
                    'dominant_race': attributes['dominant_race'],
                    'eye_redness': eye_redness,
                    'under_eye_bags': under_eye_bags,
                    'droopy_eyelids': droopy_eyelids,
                    'forehead_wrinkles': forehead_wrinkles,
                    'acne_lesions': acne_lesions
                })
            else:
                # Fallback if face location detection fails
                face_attributes = convert_to_json_serializable({
                    'age': attributes['age'],
                    'gender': attributes['gender'],
                    'dominant_emotion': attributes['dominant_emotion'],
                    'dominant_race': attributes['dominant_race'],
                    'eye_redness': None,
                    'under_eye_bags': None,
                    'droopy_eyelids': None,
                    'forehead_wrinkles': None,
                    'acne_lesions': None
                })
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
            
            face_result = {
                'name': name,
                'location': {
                    'top': top,
                    'right': right,
                    'bottom': bottom,
                    'left': left
                },
                'attributes': attributes
            }
            results.append(face_result)
        
        return convert_to_json_serializable(results)

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

    def analyze_droopy_eyelids(self, face_image, predictor_path='shape_predictor_68_face_landmarks.dat'):
        """
        Analyze droopy eyelids by measuring eye openness ratio.
        Uses facial landmarks to detect when eyelids significantly overlap the eyes.
        """
        import os
        try:
            if not os.path.exists(predictor_path):
                print(f"Predictor file not found: {predictor_path}")
                return {
                    "left_eye_openness_ratio": None,
                    "right_eye_openness_ratio": None,
                    "left_eye_droopy": False,
                    "right_eye_droopy": False,
                    "droopy_eyelids_detected": False,
                    "possible_health_conditions": ["predictor file not found"]
                }
            
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)
            rects = detector(gray, 1)
            
            if not rects:
                print("No face detected in droopy eyelid analysis.")
                return {
                    "left_eye_openness_ratio": None,
                    "right_eye_openness_ratio": None,
                    "left_eye_droopy": False,
                    "right_eye_droopy": False,
                    "droopy_eyelids_detected": False,
                    "possible_health_conditions": ["no face detected"]
                }
            
            shape = predictor(gray, rects[0])
            
            # Left eye landmarks (points 36-41)
            left_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            # Right eye landmarks (points 42-47)
            right_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
            
            def calculate_eye_openness_ratio(eye_points):
                """Calculate eye aspect ratio (EAR) to measure eye openness"""
                # Vertical distances
                vertical_1 = np.sqrt((eye_points[1][0] - eye_points[5][0])**2 + (eye_points[1][1] - eye_points[5][1])**2)
                vertical_2 = np.sqrt((eye_points[2][0] - eye_points[4][0])**2 + (eye_points[2][1] - eye_points[4][1])**2)
                
                # Horizontal distance
                horizontal = np.sqrt((eye_points[0][0] - eye_points[3][0])**2 + (eye_points[0][1] - eye_points[3][1])**2)
                
                # Eye aspect ratio
                ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
                return ear
            
            left_ear = calculate_eye_openness_ratio(left_eye_points)
            right_ear = calculate_eye_openness_ratio(right_eye_points)
            
            # Threshold for droopy eyelids (typically EAR < 0.15 indicates droopy eyelids)
            # Normal EAR is around 0.2-0.3, droopy is typically < 0.15
            droopy_threshold = 0.26
            
            left_droopy = left_ear < droopy_threshold
            right_droopy = right_ear < droopy_threshold
            
            possible_conditions = []
            if left_droopy or right_droopy:
                possible_conditions = ["Hypothyroidism", "neurological fatigue"]
            
            return {
                "left_eye_openness_ratio": float(left_ear),
                "right_eye_openness_ratio": float(right_ear),
                "left_eye_droopy": bool(left_droopy),
                "right_eye_droopy": bool(right_droopy),
                "droopy_eyelids_detected": bool(left_droopy or right_droopy),
                "possible_health_conditions": possible_conditions
            }
            
        except Exception as e:
            print(f"Error in droopy eyelid analysis: {str(e)}")
            return {
                "left_eye_openness_ratio": None,
                "right_eye_openness_ratio": None,
                "left_eye_droopy": False,
                "right_eye_droopy": False,
                "droopy_eyelids_detected": False,
                "possible_health_conditions": [f"error: {str(e)}"]
            }

    def analyze_forehead_wrinkles(self, face_image, predictor_path='shape_predictor_68_face_landmarks.dat'):
        """
        Analyze forehead wrinkles by detecting horizontal lines across the forehead.
        Uses facial landmarks to identify forehead region and image processing to detect wrinkles.
        """
        import os
        try:
            if not os.path.exists(predictor_path):
                print(f"Predictor file not found: {predictor_path}")
                return {
                    "forehead_wrinkle_count": 0,
                    "forehead_wrinkle_intensity": 0.0,
                    "forehead_wrinkles_detected": False,
                    "possible_health_conditions": ["predictor file not found"]
                }
            
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)
            rects = detector(gray, 1)
            
            if not rects:
                print("No face detected in forehead wrinkle analysis.")
                return {
                    "forehead_wrinkle_count": 0,
                    "forehead_wrinkle_intensity": 0.0,
                    "forehead_wrinkles_detected": False,
                    "possible_health_conditions": ["no face detected"]
                }
            
            shape = predictor(gray, rects[0])
            
            # Get forehead region landmarks
            # Points 17-21 are right eyebrow, 22-26 are left eyebrow
            # Points 19, 20, 23, 24 are inner eyebrow points
            left_eyebrow_top = shape.part(19)
            right_eyebrow_top = shape.part(24)
            
            # Define forehead region based on eyebrows and face width
            face_width = shape.part(16).x - shape.part(0).x  # jaw width
            forehead_height = int(face_width * 0.3)  # forehead height as proportion of face width
            
            # Forehead region coordinates
            forehead_left = max(0, shape.part(0).x)
            forehead_right = min(gray.shape[1], shape.part(16).x)
            forehead_bottom = max(left_eyebrow_top.y, right_eyebrow_top.y)
            forehead_top = max(0, forehead_bottom - forehead_height)
            
            # Extract forehead region
            forehead_roi = gray[forehead_top:forehead_bottom, forehead_left:forehead_right]
            
            if forehead_roi.size == 0:
                return {
                    "forehead_wrinkle_count": 0,
                    "forehead_wrinkle_intensity": 0.0,
                    "forehead_wrinkles_detected": False,
                    "possible_health_conditions": ["forehead region too small"]
                }
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(forehead_roi, (3, 3), 0)
            
            # Detect horizontal lines using morphological operations
            # Create horizontal kernel for detecting horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            
            # Apply morphological operations to enhance horizontal lines
            morphed = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, horizontal_kernel)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(morphed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours (potential wrinkles)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours based on aspect ratio and area
            wrinkle_lines = []
            min_wrinkle_length = forehead_roi.shape[1] * 0.2  # minimum 20% of forehead width
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter for horizontal lines (high aspect ratio, minimum length)
                if aspect_ratio > 3 and w > min_wrinkle_length and cv2.contourArea(contour) > 50:
                    wrinkle_lines.append(contour)
            
            # Calculate wrinkle intensity using edge detection
            edges = cv2.Canny(forehead_roi, 30, 100)
            
            # Focus on horizontal edges for wrinkle intensity
            sobel_x = cv2.Sobel(forehead_roi, cv2.CV_64F, 0, 1, ksize=3)  # horizontal edges
            intensity = np.mean(np.abs(sobel_x))
            
            # Normalize intensity (typical range 0-50, normalize to 0-1)
            normalized_intensity = min(intensity / 50.0, 1.0)
            
            wrinkle_count = len(wrinkle_lines)
            
            # Determine if significant wrinkles are present
            # Threshold: 2+ wrinkles OR high intensity (>0.3)
            wrinkles_detected = wrinkle_count >= 2 or normalized_intensity > 0.5
            
            possible_conditions = []
            if wrinkles_detected:
                possible_conditions = ["Cortisol elevation", "stress aging"]
            
            return {
                "forehead_wrinkle_count": int(wrinkle_count),
                "forehead_wrinkle_intensity": float(normalized_intensity),
                "forehead_wrinkles_detected": bool(wrinkles_detected),
                "possible_health_conditions": possible_conditions
            }
            
        except Exception as e:
            print(f"Error in forehead wrinkle analysis: {str(e)}")
            return {
                "forehead_wrinkle_count": 0,
                "forehead_wrinkle_intensity": 0.0,
                "forehead_wrinkles_detected": False,
                "possible_health_conditions": [f"error: {str(e)}"]
            }

    def analyze_acne_lesions(self, face_image, predictor_path='shape_predictor_68_face_landmarks.dat'):
        """
        Analyze red, inflamed papules and pustules to detect acne vulgaris.
        Uses color analysis and texture detection to identify inflammatory acne lesions.
        """
        import os
        try:
            if not os.path.exists(predictor_path):
                print(f"Predictor file not found: {predictor_path}")
                return {
                    "papule_count": 0,
                    "pustule_count": 0,
                    "total_lesions": 0,
                    "acne_severity": "none",
                    "inflammation_level": 0.0,
                    "acne_detected": False,
                    "possible_health_conditions": ["predictor file not found"]
                }
            
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)
            rects = detector(gray, 1)
            
            if not rects:
                print("No face detected in acne analysis.")
                return {
                    "papule_count": 0,
                    "pustule_count": 0,
                    "total_lesions": 0,
                    "acne_severity": "none",
                    "inflammation_level": 0.0,
                    "acne_detected": False,
                    "possible_health_conditions": ["no face detected"]
                }
            
            # Convert to different color spaces for better analysis
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            
            # Define skin regions to analyze (avoiding eyes, mouth)
            shape = predictor(gray, rects[0])

            # Build a broad face mask: full face rectangle between brows and chin
            face_left = max(0, shape.part(0).x)
            face_right = min(gray.shape[1], shape.part(16).x)
            brow_y = min(shape.part(19).y, shape.part(24).y)
            chin_y = shape.part(8).y
            face_width = face_right - face_left
            # Extend upwards from brows to include upper forehead region
            face_top = max(0, brow_y - int(0.4 * face_width))
            # Slightly below chin for coverage
            face_bottom = min(gray.shape[0], chin_y + int(0.05 * face_width))

            # Create mask for analysis regions (exclude eyes, nose, mouth later)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.rectangle(mask, (face_left, face_top), (face_right, face_bottom), 255, -1)

            # Remove eye and mouth regions
            # Eyes
            left_eye_center = ((shape.part(36).x + shape.part(39).x) // 2, (shape.part(36).y + shape.part(39).y) // 2)
            right_eye_center = ((shape.part(42).x + shape.part(45).x) // 2, (shape.part(42).y + shape.part(45).y) // 2)
            eye_radius_x = max(20, int(0.08 * face_width))
            eye_radius_y = max(12, int(0.05 * face_width))
            cv2.ellipse(mask, left_eye_center, (eye_radius_x, eye_radius_y), 0, 0, 360, 0, -1)
            cv2.ellipse(mask, right_eye_center, (eye_radius_x, eye_radius_y), 0, 0, 360, 0, -1)

            # Mouth
            mouth_center = ((shape.part(48).x + shape.part(54).x) // 2, (shape.part(48).y + shape.part(54).y) // 2)
            mouth_radius_x = max(22, int(0.12 * face_width))
            mouth_radius_y = max(12, int(0.06 * face_width))
            cv2.ellipse(mask, mouth_center, (mouth_radius_x, mouth_radius_y), 0, 0, 360, 0, -1)

            # Nose
            nose_center = (shape.part(30).x, shape.part(30).y)
            nose_radius_x = max(14, int(0.06 * face_width))
            nose_radius_y = max(10, int(0.05 * face_width))
            cv2.ellipse(mask, nose_center, (nose_radius_x, nose_radius_y), 0, 0, 360, 0, -1)

            # Apply mask to get analysis regions
            analysis_region = cv2.bitwise_and(face_image, face_image, mask=mask)
            
            # Detect red, inflamed areas using color analysis
            # Red detection in HSV space - made more sensitive for acne detection
            # Red hue ranges: 0-15 and 165-180 (expanded for better detection)
            lower_red1 = np.array([0, 30, 30])  # Lower saturation/value thresholds
            upper_red1 = np.array([15, 255, 255])  # Expanded hue range
            lower_red2 = np.array([165, 30, 30])  # Lower saturation/value thresholds
            upper_red2 = np.array([180, 255, 255])  # Expanded hue range
            
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Apply analysis region mask to red detection
            red_mask = cv2.bitwise_and(red_mask, mask)
            
            # Detect pustules (white/yellow heads) using LAB color space
            # Look for high L (lightness) and positive b (yellow)
            l_channel = lab[:,:,0]
            b_channel = lab[:,:,2]
            
            # Pustule detection: bright spots with yellow tint - made more sensitive
            pustule_mask_l = (l_channel > 160).astype(np.uint8) * 255  # Lower lightness threshold
            pustule_mask_b = (b_channel > 128).astype(np.uint8) * 255  # Lower yellow threshold
            pustule_mask = cv2.bitwise_and(pustule_mask_l, pustule_mask_b)
            pustule_mask = cv2.bitwise_and(pustule_mask, mask)
            
            # Morphological operations to clean up detection
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            pustule_mask = cv2.morphologyEx(pustule_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours for lesions
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pustule_contours, _ = cv2.findContours(pustule_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours based on size and circularity - made more sensitive
            def is_valid_lesion(contour, min_area=5, max_area=300):  # Smaller min, larger max
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    return False
                
                # Check circularity (lesions should be roughly circular) - more lenient
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    return False
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                return circularity > 0.2  # More lenient circularity requirement
            
            # Count valid papules (red lesions)
            papules = [c for c in red_contours if is_valid_lesion(c)]
            papule_count = len(papules)

            # Blob detection fallback to count many small papules that may merge/split poorly
            try:
                params = cv2.SimpleBlobDetector_Params()
                params.minThreshold = 10
                params.maxThreshold = 255
                params.filterByArea = True
                params.minArea = 8
                params.maxArea = 600
                params.filterByCircularity = True
                params.minCircularity = 0.2
                params.filterByConvexity = False
                params.filterByInertia = False
                params.filterByColor = True
                params.blobColor = 255
                detector = cv2.SimpleBlobDetector_create(params)
                # Smooth a bit to split noise from blobs
                red_mask_blur = cv2.medianBlur(red_mask, 5)
                keypoints = detector.detect(red_mask_blur)
                blob_count = len(keypoints)
                papule_count = max(papule_count, blob_count)
            except Exception as _:
                blob_count = -1
            
            # Count valid pustules (white/yellow lesions)
            pustules = [c for c in pustule_contours if is_valid_lesion(c)]
            pustule_count = len(pustules)
            
            total_lesions = papule_count + pustule_count
            
            # Calculate inflammation level based on red intensity
            if np.sum(mask) > 0:
                red_pixels = np.sum(red_mask)
                total_skin_pixels = np.sum(mask)
                inflammation_level = min(red_pixels / total_skin_pixels, 1.0)
            else:
                inflammation_level = 0.0
            
            # Determine acne severity
            if total_lesions == 0:
                severity = "none"
            elif total_lesions <= 5:
                severity = "mild"
            elif total_lesions <= 15:
                severity = "moderate"
            else:
                severity = "severe"
            
            # Detect acne vulgaris - more sensitive detection
            acne_detected = total_lesions >= 1 or inflammation_level > 0.02
            
            # Debug information
            print(f"Acne Analysis Debug:")
            print(f"  - Papules found: {papule_count}")
            print(f"  - Pustules found: {pustule_count}")
            print(f"  - Total lesions: {total_lesions}")
            print(f"  - Inflammation level: {inflammation_level:.4f}")
            print(f"  - Red contours total: {len(red_contours)}")
            print(f"  - Pustule contours total: {len(pustule_contours)}")
            print(f"  - Analysis mask area: {np.sum(mask)}")
            print(f"  - Acne detected: {acne_detected}")
            print(f"  - Blob count (fallback): {blob_count}")
            
            possible_conditions = []
            if acne_detected:
                possible_conditions = ["Acne vulgaris"]
            
            return {
                "papule_count": int(papule_count),
                "pustule_count": int(pustule_count),
                "total_lesions": int(total_lesions),
                "acne_severity": severity,
                "inflammation_level": float(inflammation_level),
                "acne_detected": bool(acne_detected),
                "possible_health_conditions": possible_conditions
            }
            
        except Exception as e:
            print(f"Error in acne lesion analysis: {str(e)}")
            return {
                "papule_count": 0,
                "pustule_count": 0,
                "total_lesions": 0,
                "acne_severity": "none",
                "inflammation_level": 0.0,
                "acne_detected": False,
                "possible_health_conditions": [f"error: {str(e)}"]
            }

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
                "left_eye_bag_darkness": float(left_darkness) if left_darkness is not None else None,
                "right_eye_bag_darkness": float(right_darkness) if right_darkness is not None else None,
                "left_relative_darkness": float(left_relative) if left_relative is not None else None,
                "right_relative_darkness": float(right_relative) if right_relative is not None else None,
                "deep_under_eye_bags": bool(left_bag or right_bag),
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
            droopy_eyelids = self.analyze_droopy_eyelids(face_image)
            forehead_wrinkles = self.analyze_forehead_wrinkles(face_image)
            acne_lesions = self.analyze_acne_lesions(face_image)
            attributes = convert_to_json_serializable({
                'age': attributes['age'],
                'gender': attributes['gender'],
                'dominant_emotion': attributes['dominant_emotion'],
                'dominant_race': attributes['dominant_race'],
                'eye_redness': eye_redness,
                'under_eye_bags': under_eye_bags,
                'droopy_eyelids': droopy_eyelids,
                'forehead_wrinkles': forehead_wrinkles,
                'acne_lesions': acne_lesions
            })
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