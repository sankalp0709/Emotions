import cv2
import numpy as np
import time
import argparse
import os
from collections import deque, Counter
from fer.fer import FER
from ai_model_adapter import EmotionModelAdapter
from event_logger import log_event
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import vlc
except Exception:
    vlc = None
try:
    import pyautogui
except Exception:
    pyautogui = None
try:
    import pymsgbox
except Exception:
    pymsgbox = None

def ear_from_landmarks(lms, w, h, idxs):
    def p(i):
        return np.array([lms[i].x * w, lms[i].y * h])
    p1, p2, p3, p4, p5, p6 = [p(i) for i in idxs]
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4) + 1e-6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="", help="Path to learning video")
    ap.add_argument("--confused_secs", type=float, default=10.0)
    ap.add_argument("--not_mood_secs", type=float, default=15.0)
    ap.add_argument("--sad_secs", type=float, default=10.0)
    ap.add_argument("--resume_lock_secs", type=float, default=120.0)
    ap.add_argument("--inference_interval_secs", type=float, default=2.5)
    ap.add_argument("--custom_model_config", type=str, default="")
    args = ap.parse_args()

    cap = cv2.VideoCapture(0)
    fer_model = FER(mtcnn=False)
    adapter = None
    if args.custom_model_config:
        try:
            adapter = EmotionModelAdapter(args.custom_model_config)
        except Exception:
            adapter = None

    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=False,
                                           output_facial_transformation_matrixes=False,
                                           num_faces=1)
    face_detector = vision.FaceLandmarker.create_from_options(options)

    player = None
    playing = False
    if args.video and vlc is not None:
        try:
            instance = vlc.Instance()
            player = instance.media_player_new()
            media = instance.media_new(args.video)
            player.set_media(media)
            player.play()
            playing = True
        except Exception:
            player = None

    def pause_video():
        nonlocal playing
        if player is not None:
            try:
                if player.is_playing():
                    player.pause()
                playing = False
            except Exception:
                pass
        elif pyautogui is not None:
            if playing:
                pyautogui.press('space')
                playing = False

    def resume_video():
        nonlocal playing
        if player is not None:
            try:
                player.play()
                playing = True
            except Exception:
                pass
        elif pyautogui is not None:
            if not playing:
                pyautogui.press('space')
                playing = True

    emotion_buf = deque(maxlen=10)
    last_pause_reason = ""
    last_action_time = 0.0
    resume_disabled_until = 0.0
    stable_label = None
    stable_count = 0
    confused_start = 0.0
    sleepy_start = 0.0
    disengaged_start = 0.0
    sad_start = 0.0
    session_start = time.time()
    last_inf_time = 0.0
    prev_time = time.time()
    drowsy_accum = 0.0
    ear_threshold = 0.21

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = face_detector.detect(mp_image)
        now = time.time()
        dt = max(0.0, now - prev_time)
        prev_time = now
        if detection_result.face_landmarks:
            lms = detection_result.face_landmarks[0]
            left_idxs = [33, 159, 160, 133, 144, 145]
            right_idxs = [362, 386, 387, 263, 373, 374]
            left_ear = ear_from_landmarks(lms, w, h, left_idxs)
            right_ear = ear_from_landmarks(lms, w, h, right_idxs)
            ear = (left_ear + right_ear) / 2.0
            if ear < ear_threshold:
                drowsy_accum += dt
            else:
                drowsy_accum = max(0.0, drowsy_accum - dt)
            cv2.putText(frame, f"EAR: {ear:.2f} Drowsy:{int(drowsy_accum)}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        faces = fer_model.detect_emotions(image_rgb)
        top_label = None
        top_score = 0.0
        if faces:
            em = faces[0]["emotions"]
            top_label = max(em, key=em.get)
            top_score = float(em[top_label])
            emotion_buf.append(top_label)
            x, y, ww, hh = faces[0]["box"]
            cv2.rectangle(frame, (x, y), (x+ww, y+hh), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {top_label} {top_score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            items = sorted(em.items(), key=lambda kv: kv[1], reverse=True)[:3]
            y0 = 80
            for i, (ln, sc) in enumerate(items):
                cv2.putText(frame, f"{ln}: {sc:.2f}", (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
        elif adapter is not None:
            roi = None
            if detection_result.face_landmarks:
                pts = detection_result.face_landmarks[0]
                xs = [int(p.x * w) for p in pts]
                ys = [int(p.y * h) for p in pts]
                x1 = max(0, min(xs))
                y1 = max(0, min(ys))
                x2 = min(w, max(xs))
                y2 = min(h, max(ys))
                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            rgb_roi = None
            if roi is not None:
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            label, score, proba = adapter.predict_with_proba(rgb_roi if rgb_roi is not None else image_rgb)
            top_label = label
            top_score = float(score)
            emotion_buf.append(top_label)
            cv2.putText(frame, f"Emotion: {top_label} {top_score:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            if proba:
                items = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)[:3]
                y0 = 80
                for i, (ln, sc) in enumerate(items):
                    cv2.putText(frame, f"{ln}: {sc:.2f}", (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)


        now = time.time()
        if top_label is not None and top_score >= 0.7 and (now - last_inf_time) >= float(args.inference_interval_secs):
            if stable_label == top_label:
                stable_count += 1
            else:
                stable_label = top_label
                stable_count = 1
                if top_label == 'confused':
                    confused_start = now
                elif top_label == 'sleepy':
                    sleepy_start = now
                elif top_label == 'disengaged':
                    disengaged_start = now
                elif top_label == 'sad':
                    sad_start = now
            last_inf_time = now

        

        now = time.time()
        if ((stable_label == 'sleepy' and stable_count >= 3 and sleepy_start and (now - sleepy_start) >= (3 * float(args.inference_interval_secs))) or (drowsy_accum >= 8.0)) and (now - last_action_time) > 1.0:
            if player is not None:
                try:
                    player.stop()
                except Exception:
                    pass
            else:
                pause_video()
            last_pause_reason = "Drowsy"
            if pymsgbox is not None:
                try:
                    pymsgbox.alert('You seem tired. Video stopped. Please wash your face and come back refreshed.')
                except Exception:
                    pass
            last_action_time = now
            resume_disabled_until = now + float(args.resume_lock_secs)
            try:
                log_event('sleepy', 'video_stopped', player, session_start, top_score)
            except Exception:
                pass
        elif (stable_label == 'confused' and stable_count >= 3 and confused_start and (now - confused_start) >= float(args.confused_secs)):
            pause_video()
            last_pause_reason = "Confused"
            if pymsgbox is not None:
                try:
                    pymsgbox.alert('It looks like this part is confusing. Your teacher will clarify this concept.')
                except Exception:
                    pass
            try:
                ts_ms = 0
                if player is not None:
                    ts_ms = player.get_time()
                with open('teacher_marks.csv','a') as f:
                    f.write(f"{int(time.time())},{ts_ms},doubt_zone\n")
            except Exception:
                pass
            last_action_time = now
            try:
                log_event('confused', 'video_paused', player, session_start, top_score)
            except Exception:
                pass
        elif (stable_label == 'disengaged' and stable_count >= 3 and disengaged_start and (now - disengaged_start) >= float(args.not_mood_secs)):
            pause_video()
            last_pause_reason = "Not in mood"
            if pymsgbox is not None:
                try:
                    pymsgbox.alert('Looks like your focus is low. Take a short break and come back.')
                except Exception:
                    pass
            last_action_time = now
            try:
                log_event('disengaged', 'video_paused', player, session_start, top_score)
            except Exception:
                pass
        elif (stable_label == 'sad' and stable_count >= 3 and sad_start and (now - sad_start) >= float(args.sad_secs)):
            pause_video()
            last_pause_reason = "Sad/Frustrated"
            if pymsgbox is not None:
                try:
                    pymsgbox.alert('Learning can feel tough sometimes. Let’s slow down and review together.')
                except Exception:
                    pass
            last_action_time = now
            try:
                log_event('sad', 'video_paused', player, session_start, top_score)
            except Exception:
                pass
        elif stable_label in ("happy","neutral") and stable_count >= 3 and (now - last_action_time) > 1.0:
            if now >= resume_disabled_until:
                resume_video()
            last_pause_reason = ""
            last_action_time = now

        status = "Playing" if playing else "Paused"
        reason = f"Reason: {last_pause_reason}" if last_pause_reason else ""
        cv2.putText(frame, f"Video: {status} {reason}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        model_str = "Model: Custom" if adapter is not None else "Model: FER"
        cv2.putText(frame, model_str, (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        remaining_lock = max(0, int(resume_disabled_until - now))
        if remaining_lock > 0:
            cv2.putText(frame, f"Resume available in {remaining_lock}s (press R)", (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            cv2.putText(frame, f"Press R to resume", (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Study Controller", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r') and now >= resume_disabled_until:
            resume_video()
            last_pause_reason = ""
            last_action_time = now

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
