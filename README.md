# Real-Time Attention Detection System

This project detects a person's attention in real-time using webcam input. It leverages OpenCV, MediaPipe (Tasks API), and a TensorFlow/Keras Neural Network model to analyze behavioral cues.

## Features
- **Real-time Face Detection**: Uses MediaPipe Tasks Vision Face Detector.
- **Head Pose Estimation**: Estimates head orientation (Forward, Down, Left, Right).
- **Hand Detection**: Tracks the number of hands visible in the frame.
- **Attention Classification**: A Neural Network classifies the user's state based on the extracted features.

## Prerequisites

- Python 3.8+
- Webcam

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/sankalp0709/Emotions.git
   cd Emotions
   ```

2. **Install Dependencies:**
   ```bash
   pip install opencv-python tensorflow scikit-learn mediapipe
   ```
   *Note: If you encounter issues with `pkg_resources`, it is usually included in `setuptools`.*

3. **Download MediaPipe Models:**
   The project requires specific MediaPipe model files. Run the provided script to download them automatically:
   ```bash
   cd Real-Time-Attention-Detection-System-main
   python download_models.py
   ```
   This will download:
   - `face_landmarker.task`
   - `hand_landmarker.task`
   - `blaze_face_short_range.tflite`

4. **Prepare the Scaler:**
   If `scaler.pkl` is missing, generate a dummy scaler (compatible with the pre-trained model structure):
   ```bash
   python create_dummy_scaler.py
   ```

## Usage

To run the real-time attention detection system:

```bash
cd Real-Time-Attention-Detection-System-main
python test11.py
```

- The system will open a webcam feed.
- It will display the detected face, pose information, and hand count.
- The predicted attention state will be shown on the screen.
- Press **`q`** to exit the application.

### Emotion Detection Only

To run a standalone emotion detector using your webcam:

```
cd Real-Time-Attention-Detection-System-main
python emotion_app.py
```

- Detects faces and overlays the top emotion label with confidence.
- Press `q` to exit.

### Study Controller (Emotion-Based Video Control)

Run a controller that pauses/resumes a learning video based on detected states:

```
cd Real-Time-Attention-Detection-System-main
python study_controller.py --video "C:\path\to\your\video.mp4"
```

- Pauses on: Drowsy (eyes closed), Confused (fear/surprise dominant), Not in mood (sad/angry/disgust dominant)
- Resumes on: Neutral/Happy and alert
- If VLC is installed, it will control playback directly; otherwise it sends a space keystroke to toggle most players.

Fixed Emotion → Action mapping:
- Doubtful/Confused → Pause video and prompt to ask teacher for clarification
- Sleepy/Drowsy → Stop video and alert the student to rest
- Not in Mood → Pause and suggest a short break
- Happy/Neutral → Continue video
- Sad/Frustrated → Pause and show a gentle encouragement prompt

### Use Your AI Builder Model

Export your classifier (ONNX, TFLite, or TensorFlow SavedModel) and create a config file:

```
{
  "model_type": "onnx",
  "model_path": "C:/path/to/your/emotion_model.onnx",
  "label_map_path": "label_map.json",
  "labels": ["happy","neutral","sad","confused","sleepy","disengaged"],
  "input_size": [224,224],
  "color_mode": "rgb",
  "normalize": true,
  "grayscale": false
}

Label map:

```
{
  "0": "happy",
  "1": "neutral",
  "2": "sad",
  "3": "confused",
  "4": "sleepy",
  "5": "disengaged"
}
```

- Index order must match model output indices; single-label classification per frame.
```

Run the controller with your model:

```
python study_controller.py --video "C:\path\to\video.mp4" --custom_model_config "C:\path\to\model_config.json"
```

The controller will use your labels and map them to the fixed actions above.

### Teacher Dashboard

- Event logging file: `events_log.jsonl` (JSON Lines, no images stored)
- Example entry: `{ "timestamp": "05:42", "emotion": "confused", "action": "video_paused" }`
- Build a simple timeline heatmap CSV:

```
python teacher_dashboard.py
```

- Output: `timeline_heatmap.csv` with per-minute counts for key emotions (confused, sleepy, disengaged, sad, neutral, happy)

### Privacy

- No webcam images are stored
- No video frames are saved
- Only minimal events are logged: `emotion`, `timestamp`, `action`
- Event file: `events_log.jsonl` (text-only JSON lines)

## Project Structure

- `test11.py`: Main entry point for the application.
- `face.py`: Handles face detection using MediaPipe Tasks.
- `hand.py`: Handles hand detection using MediaPipe Tasks.
- `posef.py`: Handles head pose estimation.
- `model2.weights.h5`: Pre-trained neural network weights.
- `create_dummy_scaler.py`: Utility to create a compatible `scaler.pkl`.
- `download_models.py`: Utility to download required MediaPipe model assets.

## Recent Updates
- **MediaPipe Migration**: The code has been updated to use the stable `mediapipe.tasks` API, replacing the deprecated `mediapipe.solutions`.
- **Model Compatibility**: Adjusted the model architecture to match the pre-trained weights (`model2.weights.h5`).
