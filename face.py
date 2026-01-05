import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Initialize detector globally to avoid reloading model
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
detector = vision.FaceDetector.create_from_options(options)

# Initialize Landmarker
landmarker_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
landmarker_options = vision.FaceLandmarkerOptions(
    base_options=landmarker_base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)

def calculate_ear(landmarks, indices):
    # indices: [p1, p2, p3, p4, p5, p6]
    # p1, p4 are horizontal
    # p2, p6 and p3, p5 are vertical
    # MediaPipe indices are 0-based.
    # Note: Landmarks is a list of normalized coordinates (x, y, z)
    
    # Helper to get numpy point
    def get_pt(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y])

    p1 = get_pt(indices[0]) # 362 (left) or 33 (right) - inner/outer corners
    p2 = get_pt(indices[1]) # 385
    p3 = get_pt(indices[2]) # 387
    p4 = get_pt(indices[3]) # 263
    p5 = get_pt(indices[4]) # 373
    p6 = get_pt(indices[5]) # 380

    # Distances
    # Vertical
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    
    # Horizontal
    h = np.linalg.norm(p1 - p4)
    
    if h == 0: return 0.0
    ear = (v1 + v2) / (2.0 * h)
    return ear

def calculate_mar(landmarks):
    # Mouth indices
    # 61: Left corner, 291: Right corner
    # 0: Upper lip top, 17: Lower lip bottom
    # 13: Upper lip bottom, 14: Lower lip top (inner mouth)
    
    # Using simple outer mouth height/width
    def get_pt(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y])

    left = get_pt(61)
    right = get_pt(291)
    top = get_pt(13) # Inner upper
    bottom = get_pt(14) # Inner lower
    
    width = np.linalg.norm(left - right)
    height = np.linalg.norm(top - bottom)
    
    if width == 0: return 0.0
    return height / width

def get_face_mesh_details(image):
    # Convert image to MediaPipe Image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    detection_result = landmarker.detect(mp_image)
    
    details = {}
    
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0] # Assuming 1 face
        
        # Left Eye Indices (MediaPipe)
        # 362, 385, 387, 263, 373, 380
        # Re-ordered for the function: p1(362), p2(385), p3(387), p4(263), p5(373), p6(380)
        # Note: 385/380 and 387/373 are pairs. 
        # Correct sequence for vertical pairs: (385, 380), (387, 373)
        # Horizontal: (362, 263)
        left_eye_indices = [362, 385, 387, 263, 373, 380]
        
        # Right Eye Indices
        # 33, 160, 158, 133, 153, 144
        # Horizontal: (33, 133)
        # Vertical pairs: (160, 144), (158, 153)
        right_eye_indices = [33, 160, 158, 133, 153, 144]
        
        left_ear = calculate_ear(landmarks, left_eye_indices)
        right_ear = calculate_ear(landmarks, right_eye_indices)
        
        avg_ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(landmarks)
        
        details['ear'] = avg_ear
        details['mar'] = mar
        
    return details


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
