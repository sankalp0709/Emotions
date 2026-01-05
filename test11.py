import cv2
import mediapipe as mp
import numpy as np
from posef import get_pose_para
from face import get_face_details, get_face_mesh_details
from hand import process_hands
from hand import draw_hand_landmarks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from fer.fer import FER

# Open the webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with a video file path
# Define the fixed mapping for categorical variables (same as during training)
label_mapping = {'forward': 0, 'down': 1, 'left': 2, 'right': 3}

# Load the pre-trained model weights and scaler
scaler = joblib.load('scaler.pkl')
weights_file = 'model2.weights.h5'

# Step 1: Define the same model architecture as during training
# Ensure the input shape matches the shape used during training
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(10,)))  # Input layer (adjust 10 to match number of features)
model.add(Dense(64, activation='relu'))  # Hidden layer 1
model.add(Dense(32, activation='relu'))  # Hidden layer 2
model.add(Dense(2, activation='softmax'))  # Output layer with 2 classes

# Load the trained model weights
model.load_weights(weights_file)

fer_model = FER(mtcnn=False)

# State variables for Sleepy/Bored detection
sleepy_frames = 0
yawn_frames = 0
bored_frames = 0

SLEEPY_THRESHOLD_EAR = 0.22  # Below this is sleepy
SLEEPY_CONSECUTIVE_FRAMES = 20 # approx 1-2 seconds

YAWN_THRESHOLD_MAR = 0.6 # Above this is yawning
YAWN_CONSECUTIVE_FRAMES = 15

BORED_LOOK_AWAY_FRAMES = 50 # approx 2-3 seconds of looking away

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # print("hey there\n")
    face_details = get_face_details(frame)
    mesh_details = get_face_mesh_details(frame)
    
    pose,pose_x,pose_y=get_pose_para(frame)
    hand_landmarks = process_hands(frame)
    no_of_hands = len(hand_landmarks)
    frame = draw_hand_landmarks(frame, hand_landmarks)
    cv2.putText(frame, f"Hands Detected: {no_of_hands}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    print(f"Number of hands:{no_of_hands}")
    print(face_details)
    print(f'pose_x:{pose_x},pose_y:{pose_y},pose:{pose}')
    
    # --- Sleepy & Bored Logic ---
    current_status_extra = ""
    color_extra = (0, 255, 0)
    
    if mesh_details:
        ear = mesh_details.get('ear', 0.3)
        mar = mesh_details.get('mar', 0.0)
        
        # Check Sleepy
        if ear < SLEEPY_THRESHOLD_EAR:
            sleepy_frames += 1
        else:
            sleepy_frames = 0
            
        if sleepy_frames > SLEEPY_CONSECUTIVE_FRAMES:
            current_status_extra = "SLEEPY"
            color_extra = (0, 0, 255)
            
        # Check Yawning (Bored)
        if mar > YAWN_THRESHOLD_MAR:
            yawn_frames += 1
        else:
            yawn_frames = 0
            
        if yawn_frames > YAWN_CONSECUTIVE_FRAMES:
            current_status_extra = "YAWNING (BORED)"
            color_extra = (0, 165, 255)

    # Check Bored (Looking away for too long)
    if pose in ['left', 'right', 'down']:
        bored_frames += 1
    else:
        bored_frames = 0
        
    if bored_frames > BORED_LOOK_AWAY_FRAMES:
        if current_status_extra == "":
            current_status_extra = "BORED (DISTRACTED)"
            color_extra = (0, 165, 255)

    if current_status_extra:
        cv2.putText(frame, f"State: {current_status_extra}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, color_extra, 2)
    # ----------------------------

    if face_details and pose is not None:

        no_of_face=1
        # print(face_details[0]['face_x'])
        sample_input=[1,face_details[0]['face_x'],face_details[0]['face_y'],face_details[0]['face_w'],face_details[0]['face_h'],face_details[0]['face_confidence'],no_of_hands,pose,pose_x,pose_y]
        if pose in label_mapping:
            sample_input[7] = label_mapping[pose]  # 'down' -> 1
        else:
            sample_input[7] = 0 # Default to forward or handle error

        # Step 4: Preprocess the input data
        # Reshape the input sample to be 2D (1 sample, n features) and scale it
        sample_input_scaled = scaler.transform(np.array(sample_input).reshape(1, -1))  # Scaling the input

        # Step 5: Make predictions
        predictions = model.predict(sample_input_scaled)
        predicted_class = np.argmax(predictions, axis=1)  # Get the class with the highest probability

        # # Step 6: Convert the predicted class back to the corresponding label
        # predicted_label = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_class[0])]

        # Step 7: Print the predicted class
        print(f"Predicted class: {predicted_class}")
        cv2.putText(frame, f"Attention: {predicted_class[0]}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        try:
            x = max(0, face_details[0]['face_x'])
            y = max(0, face_details[0]['face_y'])
            w = max(1, face_details[0]['face_w'])
            h = max(1, face_details[0]['face_h'])
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                emotion_label, emotion_score = fer_model.top_emotion(roi)
                if emotion_label is not None:
                    cv2.putText(frame, f"Emotion: {emotion_label}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        except Exception:
            pass
    else:
        cv2.putText(frame, "Face/Pose not detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with face details
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
