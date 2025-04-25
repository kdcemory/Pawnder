# Create a properly filtered emotions file
import json
import os

# Source file - use the merged file since we confirmed it has the correct distribution
source_file = "/content/drive/MyDrive/Colab Notebooks/Pawnder/Data/annotations/merged_annotations/merged_annotations.json"

# Output file for filtered entries
filtered_file = "/content/drive/MyDrive/Colab Notebooks/Pawnder/Data/interim/emotions_only.json"

# Load data
with open(source_file, 'r') as f:
    data = json.load(f)

# Filter to include only entries with emotions
emotions_only = {}
for img_id, entry in data.items():
    if "emotions" in entry and "primary_emotion" in entry["emotions"]:
        emotions_only[img_id] = entry

# Save filtered data
with open(filtered_file, 'w') as f:
    json.dump(emotions_only, f, indent=2)

print(f"Created filtered file with {len(emotions_only)} entries (out of {len(data)} total)")

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