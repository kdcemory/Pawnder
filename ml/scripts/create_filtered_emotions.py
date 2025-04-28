# create_filtered_emotions.py
# Script to create a properly filtered emotions file

import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

def create_filtered_emotions_file(source_file, output_file):
    """
    Create a file containing only entries with emotions
    
    Args:
        source_file: Path to the source annotations file (merged_annotations.json)
        output_file: Path to save the filtered file
    """
    print(f"Creating filtered emotions file")
    print(f"Source: {source_file}")
    print(f"Output: {output_file}")
    
    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"Error: Source file does not exist: {source_file}")
        return False
    
    # Load data
    try:
        with open(source_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded source file with {len(data)} entries")
    except Exception as e:
        print(f"Error loading source file: {str(e)}")
        return False
    
    # Filter to include only entries with emotions
    emotions_only = {}
    for img_id, entry in data.items():
        if "emotions" in entry and "primary_emotion" in entry["emotions"]:
            emotions_only[img_id] = entry
    
    print(f"Found {len(emotions_only)} entries with emotions")
    
    # Save filtered data
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(emotions_only, f, indent=2)
        print(f"Saved filtered file with {len(emotions_only)} entries")
    except Exception as e:
        print(f"Error saving filtered file: {str(e)}")
        return False
    
    # Verify the emotions in this file
    emotion_counts = {}
    for img_id, entry in emotions_only.items():
        emotion = entry["emotions"]["primary_emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    total = sum(emotion_counts.values())
    print(f"\nEmotion distribution in filtered file:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total * 100)
        print(f"  {emotion}: {count} ({pct:.1f}%)")
    
    # Create a visualization
    try:
        plt.figure(figsize=(12, 6))
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        
        plt.bar(emotions, counts)
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.title('Emotion Distribution in Filtered File')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save visualization
        viz_dir = os.path.dirname(output_file)
        viz_path = os.path.join(viz_dir, 'emotion_distribution.png')
        plt.savefig(viz_path)
        plt.close()
        print(f"Saved visualization to {viz_path}")
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
    
    return True

if __name__ == "__main__":
    # Default paths
    source_file = "Data/annotations/merged_annotations/merged_annotations.json"
    output_file = "Data/interim/emotions_only.json"
    
    # Allow command-line arguments
    import sys
    if len(sys.argv) > 1:
        source_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Create filtered file
    success = create_filtered_emotions_file(source_file, output_file)
    
    if success:
        print("Successfully created filtered emotions file!")
    else:
        print("Failed to create filtered emotions file.")