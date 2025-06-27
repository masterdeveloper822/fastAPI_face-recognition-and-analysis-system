from static_face_recognition_attributes_detention import FaceRecognitionAndAnalysis
from real_time_face_recognition_attributes_detention import FaceRecognitionSystem
import cv2
import time



def main():
    mode = input("Choose mode: 'static' for image processing or 'video' for real-time video: ").strip().lower()
    
    if mode == 'static':
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


    elif mode == 'video':
        # Use the FaceRecognitionSystem class for video processing
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

    else:
        print("Invalid mode. Please choose 'static' or 'video'.")

if __name__ == "__main__":
    main()