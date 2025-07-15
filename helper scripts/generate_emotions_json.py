import json
import os
from pathlib import Path

def generate_emotions_only_json(combined_annotations_path, output_path='emotions_only.json'):
    """
    Generate emotions_only.json from combined_annotations.json
    
    Args:
        combined_annotations_path: Path to the combined annotations JSON file
        output_path: Path where the emotions_only.json will be saved
    """
    
    # Load the combined annotations
    with open(combined_annotations_path, 'r') as f:
        combined_data = json.load(f)
    
    # Initialize the emotions_only dictionary
    emotions_only = {}
    
    # Process each entry in the combined annotations
    for key, data in combined_data.items():
        # Create a simplified entry for emotions_only
        emotion_entry = {}
        
        # Extract primary emotion
        if 'emotions' in data and 'primary_emotion' in data['emotions']:
            emotion_entry['emotions'] = {
                'primary_emotion': data['emotions']['primary_emotion']
            }
        elif 'primary_emotion' in data:
            # Handle case where primary_emotion might be at top level
            emotion_entry['emotions'] = {
                'primary_emotion': data['primary_emotion']
            }
        else:
            # Skip entries without emotion data
            continue
        
        # Add split information if available
        if 'split' in data:
            emotion_entry['split'] = data['split']
        else:
            # Try to infer split from source or default to 'train'
            if 'source' in data:
                if 'validation' in data['source'].lower():
                    emotion_entry['split'] = 'validation'
                elif 'test' in data['source'].lower():
                    emotion_entry['split'] = 'test'
                else:
                    emotion_entry['split'] = 'train'
            else:
                emotion_entry['split'] = 'train'  # default
        
        # Add source information
        if 'source' in data:
            emotion_entry['source'] = data['source']
        else:
            # Try to determine source from other fields
            if 'video_frames' in str(data.get('source', '')):
                emotion_entry['source'] = 'video_frames'
            elif 'video_name' in data or 'video_id' in data:
                emotion_entry['source'] = f"video_{data.get('video_id', 'unknown')}"
            else:
                emotion_entry['source'] = 'unknown'
        
        # Handle different key formats
        # For video frames like "video_6_frame_000000.png"
        if key.startswith('video_') and '_frame_' in key:
            emotions_only[key] = emotion_entry
        # For simple frame IDs like "frame_001873"
        elif key.startswith('frame_'):
            emotions_only[key] = emotion_entry
        # For YouTube/other formatted keys like "Dog Emotion Videos/test/YT_ID_314_mp4-88_jpg.rf..."
        elif 'Dog Emotion Videos' in key or 'YT_ID' in key:
            emotions_only[key] = emotion_entry
        # For any other format, include it as is
        else:
            emotions_only[key] = emotion_entry
    
    # Save the emotions_only.json
    with open(output_path, 'w') as f:
        json.dump(emotions_only, f, indent=2)
    
    print(f"Generated {output_path} with {len(emotions_only)} entries")
    
    # Print some statistics
    emotion_counts = {}
    split_counts = {}
    source_counts = {}
    
    for entry in emotions_only.values():
        # Count emotions
        if 'emotions' in entry and 'primary_emotion' in entry['emotions']:
            emotion = entry['emotions']['primary_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Count splits
        if 'split' in entry:
            split = entry['split']
            split_counts[split] = split_counts.get(split, 0) + 1
        
        # Count sources
        if 'source' in entry:
            source = entry['source']
            source_counts[source] = source_counts.get(source, 0) + 1
    
    print("\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count}")
    
    print("\nSplit distribution:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")
    
    print("\nSource distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")


# Example usage
if __name__ == "__main__":
    # Update this path to your combined_annotations.json file
    combined_annotations_path = "C:\\Users\\kelly\\Desktop\\VSC_Pawnder\\Data\\processed\\combined_annotations.json"
    
    # You can also specify a custom output path
    output_path = "C:\\Users\\kelly\\Desktop\\VSC_Pawnder\\Data\\interim\\emotions_only.json"

    # Generate the emotions_only.json
    generate_emotions_only_json(combined_annotations_path, output_path)
    
    # If you have the data in the format shown in paste.txt, use this function instead:
    def generate_from_paste_format(paste_data_path, output_path='emotions_only.json'):
        """
        Generate emotions_only.json from data in the paste.txt format
        """
        with open(paste_data_path, 'r') as f:
            paste_data = json.load(f)
        
        emotions_only = {}
        
        for key, data in paste_data.items():
            emotion_entry = {}
            
            # Extract primary emotion
            if 'emotions' in data and 'primary_emotion' in data['emotions']:
                emotion_entry['emotions'] = {
                    'primary_emotion': data['emotions']['primary_emotion']
                }
            
            # Determine split based on source or default to train
            if 'source' in data:
                if data['source'] == 'video_frames':
                    emotion_entry['split'] = 'train'  # You may need to adjust this logic
                    emotion_entry['source'] = 'video_frames'
                else:
                    emotion_entry['split'] = 'train'
                    emotion_entry['source'] = data['source']
            else:
                emotion_entry['split'] = 'train'
                emotion_entry['source'] = 'unknown'
            
            emotions_only[key] = emotion_entry
        
        with open(output_path, 'w') as f:
            json.dump(emotions_only, f, indent=2)
        
        print(f"Generated {output_path} from paste format with {len(emotions_only)} entries")
    
    # Uncomment to use with paste.txt format:
    # generate_from_paste_format("paste.txt", "emotions_only_from_paste.json")