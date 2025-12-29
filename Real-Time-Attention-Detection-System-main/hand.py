import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Initialize Hand Landmarker globally
try:
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=2,
                                           min_hand_detection_confidence=0.5)
    detector = vision.HandLandmarker.create_from_options(options)
except Exception as e:
    print(f"Error initializing HandLandmarker: {e}")
    detector = None

def process_hands(frame):
    if detector is None:
        return []

    # Convert image to MediaPipe Image
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Detect
    detection_result = detector.detect(mp_image)
    
    hand_landmarks_list = []
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Append landmark coordinates in (x, y, z)
            hand_landmarks_list.append([(lm.x, lm.y, lm.z) for lm in hand_landmarks])
            
    return hand_landmarks_list

def draw_hand_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape
    for hand in hand_landmarks:
        # Draw landmarks
        for idx, landmark in enumerate(hand):
            x, y, z = int(landmark[0] * width), int(landmark[1] * height), landmark[2]
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            # Highlight key points: WRIST and PINKY_TIP
            if idx in [0, 17]:  # WRIST and PINKY_TIP
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    return frame

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hand_landmarks = process_hands(frame)
        no_of_hands = len(hand_landmarks)
        
        # Draw detected hands and landmarks
        frame = draw_hand_landmarks(frame, hand_landmarks)

        # Display the number of hands detected
        cv2.putText(frame, f"Hands Detected: {no_of_hands}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show output
        cv2.imshow('Hand Tracking', frame)

        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
