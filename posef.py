import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Initialize Face Landmarker globally
try:
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
except Exception as e:
    print(f"Error initializing FaceLandmarker: {e}")
    detector = None

def get_pose_para(frame):
    if detector is None:
        return None, None, None

    pose = None
    pose_x = None
    pose_y = None
    
    # Define 3D model points of facial landmarks
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float32)

    # Convert image to MediaPipe Image
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Detect
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            # Extract 2D image points from landmarks
            image_points = np.array([
                (face_landmarks[1].x * frame.shape[1], face_landmarks[1].y * frame.shape[0]),   # Nose tip
                (face_landmarks[152].x * frame.shape[1], face_landmarks[152].y * frame.shape[0]), # Chin
                (face_landmarks[33].x * frame.shape[1], face_landmarks[33].y * frame.shape[0]),  # Left eye left corner
                (face_landmarks[263].x * frame.shape[1], face_landmarks[263].y * frame.shape[0]), # Right eye right corner
                (face_landmarks[61].x * frame.shape[1], face_landmarks[61].y * frame.shape[0]),   # Left Mouth corner
                (face_landmarks[291].x * frame.shape[1], face_landmarks[291].y * frame.shape[0])  # Right mouth corner
            ], dtype=np.float32)

            # Camera internals
            focal_length = frame.shape[1]
            center = (frame.shape[1] // 2, frame.shape[0] // 2)
            camera_matrix = np.array([[focal_length, 0, center[0]],
                                    [0, focal_length, center[1]],
                                    [0, 0, 1]], dtype=np.float32)

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

            # Solve for head pose
            success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
            pose_x, pose_y, _ = euler_angles.flatten()

            # Classify head pose
            if pose_y > 15:
                pose = "right"
            elif pose_y < -10:
                pose = "left"
            elif pose_x < -10:
                pose = "down"
            else:
                pose = "forward"

            # Draw the landmarks and pose
            for landmark in image_points:
                cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 3, (0, 255, 0), -1)
            cv2.putText(frame, f'Pose: {pose}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Pose_x: {pose_x:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Pose_y: {pose_y:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return pose, pose_x, pose_y

if __name__ == "__main__":
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Display the frame
        pose, pose_x, pose_y = get_pose_para(frame)
        cv2.imshow('Head Pose Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("posef.py has been imported")
