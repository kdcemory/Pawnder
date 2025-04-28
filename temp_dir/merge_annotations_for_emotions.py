# merge_annotations_for_emotions.py
# Script to merge annotations prioritizing those with emotion data

import os
import json
from pathlib import Path
import sys

# Add the project root to path to import config
sys.path.append('/content/drive/MyDrive/Colab Notebooks/Pawnder')
from config import DATA_DIRS

def merge_annotations_for_emotion_prediction():
    """Merge annotations prioritizing those with emotion data"""
    # Define file paths
    keypoints_file = os.path.join(DATA_DIRS['interim'], 'stanford_keypoints.json')
    emotions_file = os.path.join(DATA_DIRS['interim'], 'remapped_emotions_fixed.json')
    output_file = os.path.join(DATA_DIRS['merged_annotations'], 'emotion_prediction_annotations.json')
    
    # Load keypoints data
    print(f"Loading keypoints from {keypoints_file}")
    with open(keypoints_file, 'r') as f:
        keypoints_data = json.load(f)
    print(f"Loaded {len(keypoints_data)} keypoint annotations")
    
    # Load emotions data
    print(f"Loading emotions from {emotions_file}")
    with open(emotions_file, 'r') as f:
        emotions_data = json.load(f)
    print(f"Loaded {len(emotions_data)} emotion annotations")
    
    # Start with ALL emotion annotations
    merged_data = {}
    for img_id, data in emotions_data.items():
        if "emotions" in data and "primary_emotion" in data["emotions"]:
            merged_data[img_id] = {
                "emotions": data["emotions"],
                "split": data.get("split", "train"),
                "source": data.get("source", "personal")
            }
            
            # Copy over any other fields
            for key, value in data.items():
                if key not in ["emotions", "split", "source"]:
                    merged_data[img_id][key] = value
    
    # Add keypoints where available to emotion annotations
    keypoints_added = 0
    for img_id, data in keypoints_data.items():
        if img_id in merged_data and "keypoints" in data:
            merged_data[img_id]["keypoints"] = data["keypoints"]
            keypoints_added += 1
    
    # Save merged annotations
    print(f"Saving {len(merged_data)} emotion prediction annotations to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merge Results:")
    print(f"  Total annotations with emotions: {len(merged_data)}")
    print(f"  Annotations with both emotions and keypoints: {keypoints_added}")
    
    return output_file

if __name__ == "__main__":
    # If run directly, execute the merge
    output_file = merge_annotations_for_emotion_prediction()
    print(f"Merged annotations saved to: {output_file}")