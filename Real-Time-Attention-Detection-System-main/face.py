import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Initialize detector globally to avoid reloading model
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
detector = vision.FaceDetector.create_from_options(options)

def get_face_details(image):
    face_details = []
    
    # Convert image to MediaPipe Image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Detect
    detection_result = detector.detect(mp_image)
    
    if detection_result.detections:
        for i, detection in enumerate(detection_result.detections):
            bbox = detection.bounding_box
            face_x = bbox.origin_x
            face_y = bbox.origin_y
            face_w = bbox.width
            face_h = bbox.height
            face_confidence = detection.categories[0].score
            
            face_details.append({
                "face_number": i,
                "face_x": face_x,
                "face_y": face_y,
                "face_w": face_w,
                "face_h": face_h,
                "face_confidence": face_confidence
            })

            # Draw bounding box and face number on the image
            cv2.rectangle(image, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
            cv2.putText(image, f'Face {i}', (face_x, face_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(image, f'Confidence: {face_confidence:.2f}', (face_x, face_y + face_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return face_details

if __name__ == "__main__":
    # Open the webcam or video file
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face_details = get_face_details(frame)
        print(face_details)

        # Display the frame with face details
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("face.py has been imported")
