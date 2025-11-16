"""
Data Preparation Pipeline (Archive)

This script serves as an archive of the data engineering and feature extraction
pipeline used to create the 'MANIAC_benchmark_dataset.csv' from the raw
MANIAC video dataset.

This code is for archival and transparency purposes and is not intended
to be run directly as part of the main model training. The main scripts
(1_Baseline_MLP.py, 2_Temporal_BiRNN.py, 3_Champion_Model_Static_RNN.py)
all consume the final CSV file directly.

The pipeline consists of these major steps (as detailed in the paper):
1.  Temporal Downsampling (Keyframe Extraction)
2.  Predictive Windowing
3.  Kinematic and Relational Feature Engineering (using EDT)
4.  Manual Labeling
"""

import os
import cv2
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import itertools
from scipy.ndimage import center_of_mass, distance_transform_edt

# --- Configuration ---
# These would be set for each video processed
VIDEO_PATH = "/path/to/raw/video.mpg"
LABELS_TAR_PATH = "/path/to/labels.tar.gz"
EXTRACT_PATH = "/path/to/extracted_labels/"
OBJECT_ID = 10  # Example ID
HAND_ID = 16    # Example ID
WINDOW_SIZE = 10
CONTACT_THRESHOLD_PIXELS = 10

# --- Step 1: Temporal Downsampling (Keyframe Extraction) ---

def compute_importance(prev_frame, curr_frame, sharp_thresh=100, motion_thresh=20):
    """
    Scores a frame based on sharpness (Laplacian) and motion (frame diff).
    This corresponds to Section III.A (Step 1) of the paper.
    """
    if prev_frame is None:
        return 0, 0
    
    gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    motion = np.mean(cv2.absdiff(curr_gray, prev_gray))
    
    # Return normalized scores
    sharp_score = sharpness / sharp_thresh
    motion_score = motion / motion_thresh
    
    return sharp_score, motion_score

def extract_keyframes(video_path, score_threshold=1.0):
    """
    Extracts keyframes from a video that exceed sharpness and motion thresholds.
    """
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    prev_frame = None
    keyframes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sharp_score, motion_score = compute_importance(prev_frame, frame)
        total_score = (0.6 * sharp_score) + (0.4 * motion_score)

        if total_score >= score_threshold:
            keyframes.append((frame_id, frame.copy()))

        prev_frame = frame
        frame_id += 1

    cap.release()
    print(f"Extracted {len(keyframes)} keyframes from {frame_id} total frames.")
    return keyframes

# --- Step 2: Predictive Windowing ---

def create_sliding_windows(keyframes, window_size=10):
    """
    Creates predictive sliding windows from the list of keyframes.
    This corresponds to Section III.A (Step 2) of the paper.
    """
    windows = []
    for i in range(len(keyframes) - window_size):
        # Window: 10 keyframes (e.g., frames 0-9)
        window_frames = keyframes[i : i + window_size]
        # Label: The 11th keyframe (e.g., frame 10)
        label_keyframe = keyframes[i + window_size]

        frame_indices = [f[0] for f in window_frames]
        label_frame_index = label_keyframe[0]
        
        windows.append((frame_indices, label_frame_index))
    
    print(f"Created {len(windows)} sliding windows.")
    return windows

# --- Utility: Label File Mapping ---

def get_label_file_map(labels_tar_path, extract_path):
    """
    Extracts label files and creates a map from frame_id -> file_name.
    """
    os.makedirs(extract_path, exist_ok=True)
    with tarfile.open(labels_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    
    sorted_dat_files = sorted(os.listdir(extract_path))
    
    # Assumes frame 0 corresponds to file 00000_...dat, etc.
    frame_to_dat_map = {
        i: sorted_dat_files[i]
        for i in range(len(sorted_dat_files))
    }
    return frame_to_dat_map

# --- Step 3: Statistical-Kinematic Feature Engineering ---

def summarize_window_statistics(values):
    """Calculates mean, variance, and trend for a list of values."""
    if not values:
        return -1, -1, 'static'
        
    mean = np.mean(values)
    var = np.var(values)
    
    trend = 'static'
    if len(values) > 1:
        # Fit a line (y = mx + c)
        slope = np.polyfit(range(len(values)), values, 1)[0]
        if slope < -0.1: # Threshold to avoid noise
            trend = 'decreasing'
        elif slope > 0.1:
            trend = 'increasing'
            
    return mean, var, trend

def engineer_features_for_window(frame_indices, label_frame_index, label_map, label_dir, obj_id, hand_id, contact_thresh):
    """
    The core feature engineering pipeline from Section III.A (Step 3).
    """
    distances_edt = []
    speeds_centroid = []
    contact_sequence = []
    last_hand_pos = None

    for fid in frame_indices:
        dat_filename = label_map.get(fid)
        if not dat_filename:
            continue
            
        dat_path = os.path.join(label_dir, dat_filename)
        if not os.path.exists(dat_path):
            continue

        label_data = np.loadtxt(dat_path)
        obj_mask = (label_data == int(obj_id))
        hand_mask = (label_data == int(hand_id))

        if not obj_mask.any() or not hand_mask.any():
            contact_sequence.append(False)
            continue
            
        # 1. Precise Relational Distance (EDT)
        # As described in the paper, this is more robust than centroid distance.
        inv_obj_mask = ~obj_mask.astype(bool)
        dist_transform = distance_transform_edt(inv_obj_mask)
        min_distance = dist_transform[hand_mask.astype(bool)].min()
        distances_edt.append(min_distance)

        # 2. Kinematic Motion Cues (Centroid-based)
        hand_centroid = center_of_mass(hand_mask)
        if last_hand_pos is not None:
            speed = np.linalg.norm(np.array(hand_centroid) - np.array(last_hand_pos))
            speeds_centroid.append(speed)
        last_hand_pos = hand_centroid
        
        # 3. Sophisticated Contact Features
        is_contact = (min_distance < contact_thresh)
        contact_sequence.append(is_contact)

    # --- Aggregate all statistics for the 10-frame window ---
    
    if not distances_edt: # Skip if window had no valid hand/object pairs
        return None

    # Aggregate Distances
    mean_dist, var_dist, trend_dist = summarize_window_statistics(distances_edt)
    
    # Aggregate Speeds
    mean_speed, var_speed, _ = summarize_window_statistics(speeds_centroid)
    # The 'acceleration' feature mentioned in the notebook
    std_speed = np.std(speeds_centroid) if speeds_centroid else -1 
    
    # Aggregate Contact
    contact_count = np.sum(contact_sequence)
    
    # Find longest un-interrupted contact sequence
    contact_duration = 0
    if contact_count > 0:
        groups = [list(g) for k, g in itertools.groupby(contact_sequence) if k]
        if groups:
            contact_duration = max(len(g) for g in groups)

    feature_vector = {
        "sample_id": f"{frame_indices[0]}_{frame_indices[-1]}",
        "frame_start": frame_indices[0],
        "frame_end": frame_indices[-1],
        "label_frame": label_frame_index,
        "avg_distance": mean_dist,
        "var_distance": var_dist,
        "distance_trend": trend_dist,
        "contact_count": contact_count,
        "contact_duration": contact_duration,
        "avg_speed": mean_speed,
        "acceleration": std_speed, # Corresponds to 'std dev of speed'
        # Other features like 'avg_angle' were in the notebook but omitted
        # from the final paper's description for brevity.
    }
    
    return feature_vector

# --- Step 4: Manual Labeling (Example UI) ---

def manual_labeling_interface(df_features, video_path):
    """
    This is a conceptual, non-runnable example of the Colab-based
    manual labeling UI used in the R&D notebook.
    """
    print("--- Starting Manual Labeling ---")
    label_options = ['approaching', 'grabbing', 'holding', 'releasing', 'unknown']
    labeled_data = []

    for _, row in df_features.iterrows():
        label_fid = int(row['label_frame'])
        
        # --- In Colab, this would display the frame ---
        # cap = cv2.VideoCapture(video_path)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, label_fid)
        # ret, frame = cap.read()
        # if ret:
        #     cv2_imshow(frame) # Colab's display function
        # cap.release()
        # ---
        
        print(f"\nLabeling window {row['sample_id']} (predicting frame {label_fid})")
        print("Features:")
        print(f"  Avg Dist: {row['avg_distance']:.1f}, Trend: {row['distance_trend']}")
        print(f"  Contact Count: {row['contact_count']}, Duration: {row['contact_duration']}")
        print(f"  Avg Speed: {row['avg_speed']:.1f}, Acceleration: {row['acceleration']:.1f}")
        
        print("Please assign a label (1-5):")
        for i, opt in enumerate(label_options, 1):
            print(f"  {i}) {opt}")
        
        # In a real script, this would be a UI input
        # user_input = input("Enter number: ") 
        # label = label_options[int(user_input) - 1]
        label = "example_label" # Placeholder
        # ---
        
        row['label'] = label
        labeled_data.append(row)

    return pd.DataFrame(labeled_data)

# --- Main Archival Execution Flow ---

def run_pipeline_for_one_video(video_path, labels_tar_path, extract_path, obj_id, hand_id):
    """
    Conceptual function showing how one video is processed.
    This would be run in a loop for all videos in the MANIAC dataset.
    """
    
    # 1. Get keyframes
    keyframes = extract_keyframes(video_path)
    
    # 2. Get sliding windows
    windows = create_sliding_windows(keyframes, window_size=WINDOW_SIZE)
    
    # 3. Get label file map
    label_map = get_label_file_map(labels_tar_path, extract_path)
    
    # 4. Engineer features for all windows
    all_features = []
    for frame_indices, label_frame_index in windows:
        features = engineer_features_for_window(
            frame_indices, label_frame_index, label_map,
            extract_path, obj_id, hand_id, CONTACT_THRESHOLD_PIXELS
        )
        if features:
            all_features.append(features)
    
    df_features = pd.DataFrame(all_features)
    
    # 5. Manually label the features
    # df_labeled = manual_labeling_interface(df_features, video_path)
    
    # 6. Save the labeled features for this video
    # df_labeled.to_csv(f"labeled_features_{video_name}.csv", index=False)
    
    print("--- Pipeline complete for one video (Example) ---")
    # In the real project, all individual CSVs are concatenated
    # to create the final 'MANIAC_benchmark_dataset.csv'


if __name__ == "__main__":
    print("This is an archival script demonstrating the data preparation pipeline.")
    print("It is not intended for direct execution.")
    # Example conceptual run:
    # run_pipeline_for_one_video(VIDEO_PATH, LABELS_TAR_PATH, EXTRACT_PATH, OBJECT_ID, HAND_ID)
