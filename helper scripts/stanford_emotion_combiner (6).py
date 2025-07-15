import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StanfordCombiner")

class StanfordEmotionCombiner:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = "C:\\Users\\kelly\\Documents\\GitHub\\Pawnder"
        else:
            self.base_dir = base_dir

        # Define paths based on your structure
        self.stanford_pose_dir = self.base_dir / "Data/raw/stanford_dog_pose"
        self.emotion_annotations_dir = self.base_dir / "Data/raw/stanford_annotations"
        self.output_dir = self.base_dir / "Data/processed/stanford_combined"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current emotion class mapping
        self.emotion_classes = {
            0: "Happy/Playful",
            1: "Relaxed", 
            2: "Curiosity/Alertness",
            3: "Stressed",
            4: "Fearful/Anxious", 
            5: "Aggressive/Threatening",
            6: "Submissive/Appeasement"
        }
        
        # Mapping from old emotion names to new standard names
        self.emotion_mapping = {
            # Current format (train set)
            "Happy/Playful": "Happy/Playful",
            "Relaxed": "Relaxed",
            "Curiosity/Alertness": "Curiosity/Alertness", 
            "Stressed": "Stressed",
            "Fearful/Anxious": "Fearful/Anxious",
            "Aggressive/Threatening": "Aggressive/Threatening",
            "Submissive/Appeasement": "Submissive/Appeasement",
            
            # Old format mappings (validation set)
            "Happy or Playful": "Happy/Playful",
            "Submissive": "Submissive/Appeasement",
            "Excited": "Happy/Playful",
            "Drowsy or Bored": "Relaxed",
            "Curious or Confused": "Curiosity/Alertness",
            "Confident or Alert": "Curiosity/Alertness",
            "Jealous": "Stressed",
            "Frustrated": "Stressed",
            "Unsure or Uneasy": "Fearful/Anxious",
            "Possessive, Territorial, Dominant": "Aggressive/Threatening",
            "Fear or Aggression": "Fearful/Anxious",
            "Pain": "Stressed"
        }
        
        # Reverse mapping: standard name to class ID
        self.name_to_id = {name: id for id, name in self.emotion_classes.items()}
    
    def load_yolo_keypoints(self, labels_dir):
        """Load keypoints from YOLO format label files"""
        keypoints = {}
        
        if not labels_dir.exists():
            logger.warning(f"Labels directory not found: {labels_dir}")
            return keypoints
        
        label_files = list(labels_dir.glob("*.txt"))
        logger.info(f"Loading {len(label_files)} YOLO keypoint files from {labels_dir}")
        
        for label_file in tqdm(label_files, desc="Loading YOLO keypoints"):
            try:
                image_id = label_file.stem
                
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 3:  # At least class_id + bbox
                        class_id = int(parts[0])
                        bbox_x = float(parts[1])
                        bbox_y = float(parts[2])
                        bbox_w = float(parts[3]) if len(parts) > 3 else 0
                        bbox_h = float(parts[4]) if len(parts) > 4 else 0
                        
                        # Extract keypoints (start from index 5)
                        keypoint_data = []
                        if len(parts) > 5:
                            kp_values = [float(x) for x in parts[5:]]
                            # Group into triplets (x, y, visibility)
                            for i in range(0, len(kp_values), 3):
                                if i + 2 < len(kp_values):
                                    keypoint_data.append({
                                        "x": kp_values[i],
                                        "y": kp_values[i + 1],
                                        "visibility": kp_values[i + 2]
                                    })
                        
                        keypoints[image_id] = {
                            "class_id": class_id,
                            "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                            "keypoints": keypoint_data
                        }
                        
            except Exception as e:
                logger.warning(f"Error processing {label_file}: {e}")
        
        logger.info(f"Loaded keypoints for {len(keypoints)} images")
        return keypoints
    
    def load_emotion_annotations(self, json_file):
        """Load emotion annotations from CVAT COCO format"""
        emotions = {}
        
        if not json_file.exists():
            logger.warning(f"Emotion annotations not found: {json_file}")
            return emotions
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loading emotion annotations from {json_file}")
            
            if 'annotations' in data and 'images' in data:
                # CVAT COCO format
                annotations = data['annotations']
                images = {img['id']: img for img in data['images']}
                
                for ann in annotations:
                    image_id = ann['image_id']
                    if image_id in images:
                        filename = images[image_id]['file_name']
                        image_key = Path(filename).stem
                        
                        # Extract emotion from attributes
                        attributes = ann.get('attributes', {})
                        primary_emotion_raw = attributes.get('Primary Emotion', attributes.get('primary_emotion'))
                        
                        # Map old emotion names to new standard names
                        primary_emotion_mapped = None
                        emotion_class_id = None
                        
                        if primary_emotion_raw:
                            # Apply emotion mapping
                            primary_emotion_mapped = self.emotion_mapping.get(primary_emotion_raw, primary_emotion_raw)
                            # Get class ID from mapped emotion
                            emotion_class_id = self.name_to_id.get(primary_emotion_mapped)
                            
                            if emotion_class_id is None:
                                logger.warning(f"Unknown emotion after mapping: '{primary_emotion_raw}' -> '{primary_emotion_mapped}'")
                        
                        # Extract behavioral indicators
                        behavioral_indicators = {}
                        for key, value in attributes.items():
                            if key != 'Primary Emotion' and key != 'primary_emotion':
                                behavioral_indicators[key] = value
                        
                        emotions[image_key] = {
                            "emotion_class": emotion_class_id,
                            "emotion_name": primary_emotion_mapped,
                            "emotion_original": primary_emotion_raw,  # Keep original for reference
                            "confidence": 1.0,  # CVAT doesn't provide confidence
                            "behavioral_indicators": behavioral_indicators,
                            "filename": filename,
                            "annotation_id": ann['id'],
                            "bbox": ann.get('bbox', [])  # CVAT bbox if available
                        }
            
            logger.info(f"Loaded emotion annotations for {len(emotions)} images")
            
            # Print emotion distribution for debugging
            emotion_dist = {}
            for em_data in emotions.values():
                emotion = em_data['emotion_name'] or 'unknown'
                emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
            
            logger.info(f"Emotion distribution: {emotion_dist}")
            
        except Exception as e:
            logger.error(f"Error loading emotion annotations: {e}")
        
        return emotions
    
    def combine_annotations(self, keypoints, emotions, split_name):
        """Combine keypoint and emotion annotations"""
        combined = []
        
        # Find intersection of images with both keypoints and emotions
        common_images = set(keypoints.keys()) & set(emotions.keys())
        logger.info(f"Found {len(common_images)} images with both keypoints and emotions in {split_name}")
        
        for image_id in tqdm(common_images, desc=f"Combining {split_name} annotations"):
            kp_data = keypoints[image_id]
            em_data = emotions[image_id]
            
            # Count visible keypoints
            visible_kps = sum(1 for kp in kp_data['keypoints'] if kp['visibility'] > 0)
            
            combined_entry = {
                "image_id": image_id,
                "filename": em_data.get('filename', f"{image_id}.jpg"),
                "split": split_name,
                
                # Keypoint data
                "has_keypoints": len(kp_data['keypoints']) > 0,
                "keypoint_count": len(kp_data['keypoints']),
                "visible_keypoints": visible_kps,
                "bbox": kp_data['bbox'],
                
                # Emotion data (mapped to standard format)
                "emotion_class": em_data.get('emotion_class'),
                "emotion_name": em_data.get('emotion_name'),  # Standardized name
                "emotion_original": em_data.get('emotion_original'),  # Original annotation
                "confidence": em_data.get('confidence', 1.0),
                
                # Behavioral indicators from CVAT
                "behavioral_indicators": em_data.get('behavioral_indicators', {}),
                
                # Raw data for training
                "keypoints_raw": kp_data['keypoints'],
                "annotation_id": em_data.get('annotation_id')
            }
            
            combined.append(combined_entry)
        
        return combined
    
    def save_combined_annotations(self, combined_data, split_name):
        """Save combined annotations in multiple formats"""
        
        # Save as CSV for easy viewing
        df = pd.DataFrame(combined_data)
        csv_path = self.output_dir / f"{split_name}_combined.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")
        
        # Save as JSON for training
        json_path = self.output_dir / f"{split_name}_combined.json"
        with open(json_path, 'w') as f:
            json.dump(combined_data, f, indent=2)
        logger.info(f"Saved JSON: {json_path}")
        
        # Create summary
        summary = {
            "split": split_name,
            "total_samples": len(combined_data),
            "emotion_distribution": {},
            "keypoint_stats": {
                "avg_visible_keypoints": np.mean([item["visible_keypoints"] for item in combined_data]),
                "min_visible_keypoints": min([item["visible_keypoints"] for item in combined_data]),
                "max_visible_keypoints": max([item["visible_keypoints"] for item in combined_data])
            }
        }
        
        # Count emotion distribution
        for item in combined_data:
            emotion = item["emotion_name"] or f"class_{item['emotion_class']}"
            summary["emotion_distribution"][emotion] = summary["emotion_distribution"].get(emotion, 0) + 1
        
        return summary
    
    def process_all_splits(self):
        """Process both train and validation splits"""
        summaries = {}
        
        # Process training set
        train_keypoints = self.load_yolo_keypoints(
            self.stanford_pose_dir / "dog-pose" / "train" / "labels"
        )
        train_emotions = self.load_emotion_annotations(
            self.emotion_annotations_dir / "instances_train.json"
        )
        
        if train_keypoints and train_emotions:
            train_combined = self.combine_annotations(train_keypoints, train_emotions, "train")
            summaries["train"] = self.save_combined_annotations(train_combined, "train")
        
        # Process validation set
        val_keypoints = self.load_yolo_keypoints(
            self.stanford_pose_dir / "dog-pose" / "val" / "labels"
        )
        val_emotions = self.load_emotion_annotations(
            self.emotion_annotations_dir / "instances_Validation.json"
        )
        
        if val_keypoints and val_emotions:
            val_combined = self.combine_annotations(val_keypoints, val_emotions, "validation")
            summaries["validation"] = self.save_combined_annotations(val_combined, "validation")
        
        # Save overall summary
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summaries, f, indent=2)
        
        logger.info(f"Processing complete! Summary saved to {summary_path}")
        
        # Print summary
        for split, summary in summaries.items():
            print(f"\n{split.upper()} SET SUMMARY:")
            print(f"  Total samples: {summary['total_samples']}")
            print(f"  Avg visible keypoints: {summary['keypoint_stats']['avg_visible_keypoints']:.1f}")
            print(f"  Emotion distribution:")
            for emotion, count in summary['emotion_distribution'].items():
                print(f"    {emotion}: {count}")

# Usage
if __name__ == "__main__":
    base_dir = "C:\\Users\\kelly\\Documents\\GitHub\\Pawnder"
    
    combiner = StanfordEmotionCombiner(base_dir)
    combiner.process_all_splits()