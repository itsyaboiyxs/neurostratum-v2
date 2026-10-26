import sys
import os
import urllib.request
import cv2
import pandas as pd
import mediapipe as mp

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = "pose_landmarker_heavy.task"

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe PoseLandmarker heavy model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def process_video_to_landmarks(video_path, output_csv):
    download_model_if_needed()
    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    data = []
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = int(frame_idx * 1000.0 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                for idx, lm in enumerate(landmarks):
                    data.append({
                        "frame": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "landmark_index": idx,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    })
            frame_idx += 1

    cap.release()
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} landmark records to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python python/video_to_landmarks.py <input_video.mp4> <output.csv>")
        sys.exit(1)
    process_video_to_landmarks(sys.argv[1], sys.argv[2])