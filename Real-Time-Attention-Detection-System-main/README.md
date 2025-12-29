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
