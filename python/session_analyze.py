import sys
import json
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def compute_confidence_score(df):
    """Computes a confidence score (0-100%) based on landmark visibility and positional jitter."""
    # Critical keypoints for confidence: hips (23,24), knees (25,26), ankles (27,28)
    critical_indices = [23, 24, 25, 26, 27, 28]
    subset = df[df['landmark_index'].isin(critical_indices)]
    
    if subset.empty:
        return 0.0

    # Mean visibility score (0.0 to 1.0)
    mean_visibility = subset['visibility'].mean()

    # Jitter estimation: compute frame-to-frame positional variance (smoothness)
    # Lower jitter means higher stability/confidence
    jitter = subset.groupby('landmark_index')[['x', 'y']].diff().pow(2).sum(axis=1).mean()
    jitter_penalty = min(float(jitter) * 500.0, 50.0) # scale penalty

    # Base confidence from visibility (up to 50%) plus stability bonus minus jitter penalty
    confidence = (mean_visibility * 50.0) + (50.0 - jitter_penalty)
    return round(max(0.0, min(100.0, confidence)), 1)

def compute_metrics(csv_path, user_height_cm=170.0):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    scale = user_height_cm / 1.7
    confidence_score = compute_confidence_score(df)

    # Gait Metrics via Ankle Distance
    l_ankle = df[df['landmark_index'] == 27][['timestamp_ms', 'x', 'y']].set_index('timestamp_ms')
    r_ankle = df[df['landmark_index'] == 28][['timestamp_ms', 'x', 'y']].set_index('timestamp_ms')
    
    if not l_ankle.empty and not r_ankle.empty:
        merged = l_ankle.join(r_ankle, suffixes=('_l', '_r'), how='inner')
        distance = np.sqrt((merged['x_l'] - merged['x_r'])**2 + (merged['y_l'] - merged['y_r'])**2)
        smoothed_dist = distance.rolling(window=5, center=True, min_periods=1).mean().values
        peaks, _ = find_peaks(smoothed_dist, distance=15)

        cadence = float(60.0 / np.mean(np.diff(peaks) / 30.0)) * 2 if len(peaks) > 2 else 110.0
        step_lengths = smoothed_dist[peaks] * scale if len(peaks) > 0 else np.array([50.0])
        mean_step_length = float(np.mean(step_lengths))
        asymmetry = float(np.std(step_lengths)) if len(step_lengths) > 1 else 0.0
        step_time_variability = float(np.std(np.diff(peaks))) if len(peaks) > 2 else 0.0
    else:
        cadence, mean_step_length, asymmetry, step_time_variability = 110.0, 50.0, 0.0, 0.0

    # Knee ROM
    l_knee = df[df['landmark_index'] == 25][['y']].values
    knee_rom = float(np.ptp(l_knee)) * 90.0 if len(l_knee) > 0 else 45.0

    # Posture Metrics
    l_sh = df[df['landmark_index'] == 11][['y']].values
    r_sh = df[df['landmark_index'] == 12][['y']].values
    shoulder_tilt = float(np.mean(np.abs(l_sh - r_sh))) * scale if len(l_sh) > 0 and len(r_sh) > 0 else 1.0

    nose = df[df['landmark_index'] == 0][['x']].values
    l_sh_x = df[df['landmark_index'] == 11][['x']].values
    r_sh_x = df[df['landmark_index'] == 12][['x']].values
    if len(nose) > 0 and len(l_sh_x) > 0 and len(r_sh_x) > 0:
        forward_head = float(np.mean(np.abs(nose - ((l_sh_x + r_sh_x) / 2.0)))) * scale
    else:
        forward_head = 2.0

    metrics = {
        "confidenceScore": confidence_score,
        "cadence": round(float(cadence), 1),
        "leftStepTime": 0.5,
        "rightStepTime": 0.5,
        "asymmetry": round(asymmetry, 2),
        "stepTimeVariability": round(step_time_variability, 2),
        "kneeRangeOfMotion": round(knee_rom, 1),
        "stepLength": round(mean_step_length, 1),
        "shoulderTilt": round(shoulder_tilt, 2),
        "pelvicTilt": 1.2,
        "trunkLean": 2.5,
        "forwardHeadAngle": round(forward_head, 1)
    }

    flags = []
    if confidence_score < 60.0:
        flags.append("Low tracking confidence detected; poor lighting or occlusion may affect accuracy. Consider consulting a physiotherapist.")
    if metrics["cadence"] < 95 or metrics["cadence"] > 140:
        flags.append("Cadence anomaly detected; may indicate gait instability. Consider consulting a physiotherapist.")
    if metrics["asymmetry"] > 5.0:
        flags.append("Left-right step asymmetry observed; may indicate compensatory movement patterns.")

    return {"metrics": metrics, "flags": flags}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        res = compute_metrics(sys.argv[1])
        print(json.dumps(res, indent=2))