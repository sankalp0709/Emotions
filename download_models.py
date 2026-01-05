import requests
import os

models = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
}

for name, url in models.items():
    if not os.path.exists(name):
        print(f"Downloading {name}...")
        try:
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                with open(name, 'wb') as f:
                    f.write(r.content)
                print(f"Downloaded {name}")
            else:
                print(f"Failed to download {name}: Status {r.status_code}")
        except Exception as e:
            print(f"Failed to download {name}: {e}")
    else:
        print(f"{name} already exists.")
