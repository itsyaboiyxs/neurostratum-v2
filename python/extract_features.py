import os
import glob
import pandas as pd
import numpy as np

def extract_features_from_csv(csv_path, label, participant_id):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if 'timestamp_ms' not in df.columns:
        return []

    # Group into 5-second (5000 ms) windows
    df['window'] = df['timestamp_ms'] // 5000
    features_list = []

    for window_id, group in df.groupby('window'):
        ankles = group[group['landmark_index'].isin([27, 28])]
        knees = group[group['landmark_index'].isin([25, 26])]
        
        mean_vis = group['visibility'].mean() if 'visibility' in group.columns else 1.0
        
        feature_row = {
            "participant": participant_id,
            "label": label, # e.g., 'good' or 'weird'
            "window_id": window_id,
            "mean_visibility": mean_vis,
            "ankle_x_spread": ankles['x'].std() if not ankles.empty else 0.0,
            "knee_y_spread": knees['y'].std() if not knees.empty else 0.0,
        }
        features_list.append(feature_row)

    return features_list

def process_all_training_videos():
    # Ensure training_videos directory exists
    os.makedirs("training_videos", exist_ok=True)
    
    # Scan for CSV files inside training_videos/
    csv_files = glob.glob("training_videos/*.csv")
    
    # Also check if user placed any training CSVs directly in root directory by mistake
    root_csvs = [f for f in glob.glob("*.csv") if "_" in f and f not in ["landmarks.csv", "features.csv"]]
    for rc in root_csvs:
        target_path = os.path.join("training_videos", rc)
        if not os.path.exists(target_path):
            os.rename(rc, target_path)
            print(f"Moved {rc} into training_videos/ folder.")

    csv_files = glob.glob("training_videos/*.csv")
    all_features = []

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        parts = filename.replace(".csv", "").split("_")
        if len(parts) >= 2:
            label = parts[0]
            participant_id = parts[1]
            
            features = extract_features_from_csv(csv_file, label, participant_id)
            all_features.extend(features)

    if all_features:
        features_df = pd.DataFrame(all_features)
        features_df.to_csv("features.csv", index=False)
        print(f"Extracted {len(features_df)} feature windows to features.csv")
    else:
        print("No training CSV files found in training_videos/ directory. Name them like label_participant_n.csv (e.g., weird_p1_1.csv).")

if __name__ == "__main__":
    process_all_training_videos()