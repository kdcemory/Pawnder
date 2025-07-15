"""
Fixed Training Script - Enhanced Behavioral Data Loading

Key fixes:
1. Improved behavior matrix loading with fallback options
2. Better behavioral data extraction and conversion
3. Enhanced data generator with proper behavior matching
4. Debugging output for behavioral data issues
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FixedTraining")

class FixedDogEmotionWithBehaviors:
    """
    Fixed version of DogEmotionWithBehaviors with enhanced behavioral data handling
    """
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = "C:/Users/kelly/Documents/GitHub/Pawnder"
        else:
            self.base_dir = base_dir
            
        self.processed_dir = os.path.join(self.base_dir, "Data", "processed")
        self.matrix_dir = os.path.join(self.base_dir, "Data", "Matrix")
        self.model_dir = os.path.join(self.base_dir, "Models")
        
        # Enhanced class mapping
        self.class_name_mapping = {
            "Happy": "Happy/Playful",
            "Relaxed": "Relaxed",
            "Submissive": "Submissive/Appeasement",
            "Curiosity": "Curiosity/Alertness",
            "Stressed": "Stressed",
            "Fearful": "Fearful/Anxious",
            "Aggressive": "Aggressive/Threatening",
            "Happy/Playful": "Happy/Playful",
            "Submissive/Appeasement": "Submissive/Appeasement",
            "Curiosity/Alertness": "Curiosity/Alertness",
            "Fearful/Anxious": "Fearful/Anxious",
            "Aggressive/Threatening": "Aggressive/Threatening",
        }
        
        self.model = None
        self.class_names = []
        self.behavior_columns = []
        self.behavior_matrix = None
        
        print(f"Fixed trainer initialized with base directory: {self.base_dir}")
    
    def load_behavior_matrix_enhanced(self, debug=True):
        """
        Enhanced behavior matrix loading with multiple fallback options
        """
        if debug:
            logger.info("Loading behavior matrix with enhanced fallback...")
        
        # Try multiple possible paths and formats
        potential_paths = [
            os.path.join(self.matrix_dir, "use_behavior_matrix.json"),
            os.path.join(self.matrix_dir, "behavior_matrix.json"),
            os.path.join(self.matrix_dir, "primary_behavior_matrix.json"),
            os.path.join(self.base_dir, "use_behavior_matrix.json"),
            os.path.join(self.base_dir, "behavior_matrix.json"),
            "use_behavior_matrix.json",
            "behavior_matrix.json"
        ]
        
        for matrix_path in potential_paths:
            if os.path.exists(matrix_path):
                if debug:
                    logger.info(f"Found behavior matrix at: {matrix_path}")
                
                try:
                    with open(matrix_path, 'r') as f:
                        matrix_data = json.load(f)
                    
                    # Convert to standardized format
                    self.behavior_matrix = self._convert_matrix_format(matrix_data)
                    
                    if debug:
                        logger.info(f"Successfully loaded behavior matrix:")
                        logger.info(f"  - Emotions: {len(self.behavior_matrix.get('emotions', []))}")
                        logger.info(f"  - Behaviors: {len(self.behavior_matrix.get('behaviors', {}))}")
                    
                    # Set behavior columns
                    self.behavior_columns = list(self.behavior_matrix.get('behaviors', {}).keys())
                    
                    return True
                    
                except Exception as e:
                    if debug:
                        logger.warning(f"Error loading matrix from {matrix_path}: {e}")
                    continue
        
        if debug:
            logger.warning("No behavior matrix found, creating default features")
        
        # Create default behavior features if no matrix found
        self._create_default_behaviors()
        return False
    
    def _convert_matrix_format(self, raw_matrix):
        """
        Convert behavior matrix to standardized format
        """
        # Handle different input formats
        if "behavioral indicators" in raw_matrix:
            # Format from use_behavior_matrix.json
            emotions_key = "primary emotions" if "primary emotions" in raw_matrix else "emotions"
            emotions = raw_matrix.get(emotions_key, [])
            behaviors = raw_matrix["behavioral indicators"]
            
        elif "behavioral_states" in raw_matrix and "behavior_categories" in raw_matrix:
            # Format from primary_behavior_matrix.json
            states = raw_matrix["behavioral_states"]
            categories = raw_matrix["behavior_categories"]
            
            emotions = [state["name"] for state in states]
            
            # Convert to behaviors format
            behaviors = {}
            for category in categories:
                for behavior in category.get("behaviors", []):
                    behavior_id = behavior["id"]
                    state_mapping = {}
                    
                    for state in states:
                        state_id = state["id"]
                        state_name = state["name"]
                        
                        if "state_mapping" in behavior and state_id in behavior["state_mapping"]:
                            value = behavior["state_mapping"][state_id]
                            state_mapping[state_name] = value
                        else:
                            state_mapping[state_name] = 0
                    
                    behaviors[behavior_id] = state_mapping
            
        elif "emotions" in raw_matrix and "behaviors" in raw_matrix:
            # Already in standard format
            emotions = raw_matrix["emotions"]
            behaviors = raw_matrix["behaviors"]
            
        else:
            # Unknown format, use defaults
            emotions = [
                "Happy/Playful", "Relaxed", "Submissive/Appeasement",
                "Curiosity/Alertness", "Stressed", "Fearful/Anxious",
                "Aggressive/Threatening"
            ]
            behaviors = {}
        
        return {
            "emotions": emotions,
            "behaviors": behaviors
        }
    
    def _create_default_behaviors(self):
        """
        Create default behavior features if no matrix is available
        """
        default_behaviors = [
            "relaxed_open_mouth", "tight_closed_mouth", "bared_teeth",
            "high_and_stiff_tail", "low_and_tucked_tail", "neutral_tail_with_gentle_wag",
            "forward_erect_ears", "pinned_back_ears", "neutral_relaxed_ears",
            "soft_eyes", "hard_stare", "whale_eyes",
            "relaxed_muscles", "tense_muscles", "play_bow"
        ]
        
        emotions = [
            "Happy/Playful", "Relaxed", "Submissive/Appeasement",
            "Curiosity/Alertness", "Stressed", "Fearful/Anxious",
            "Aggressive/Threatening"
        ]
        
        # Create simple mappings
        behaviors = {}
        for behavior in default_behaviors:
            behaviors[behavior] = {emotion: 0 for emotion in emotions}
        
        self.behavior_matrix = {
            "emotions": emotions,
            "behaviors": behaviors
        }
        
        self.behavior_columns = default_behaviors
        logger.info(f"Created {len(default_behaviors)} default behavior features")
    
    def normalize_behavior_name(self, behavior_name):
        """
        Consistent behavior name normalization
        """
        normalized = behavior_name.lower()
        normalized = normalized.replace(' ', '_')
        normalized = normalized.replace('/', '_')
        normalized = normalized.replace('-', '_')
        normalized = normalized.replace('(', '').replace(')', '')
        normalized = normalized.replace('&', 'and')
        # Remove extra underscores
        while '__' in normalized:
            normalized = normalized.replace('__', '_')
        return normalized.strip('_')
    
    def load_behavior_data_enhanced(self, annotations, debug=True):
        """
        Enhanced behavior data loading with better debugging
        """
        if not annotations:
            logger.warning("No annotations provided for behavior extraction")
            return {}, self.behavior_columns
        
        behavior_data = {}
        
        # Count annotations with behavioral data
        annotations_with_behaviors = 0
        total_behavior_instances = 0
        behavior_type_counts = defaultdict(int)
        
        logger.info(f"Processing {len(annotations)} annotations for behavioral data...")
        
        # Get first annotation to check structure
        first_key = next(iter(annotations))
        first_entry = annotations[first_key]
        
        if debug:
            logger.info(f"Sample annotation structure: {list(first_entry.keys())}")
        
        # Process each annotation
        for img_id, data in annotations.items():
            behavior_values = []
            found_behaviors = 0
            
            # Initialize with zeros
            for behavior_name in self.behavior_columns:
                value = 0.0
                
                # Try to find this behavior in the annotation
                if "behavioral_indicators" in data:
                    indicators = data["behavioral_indicators"]
                    
                    # Try different possible keys for this behavior
                    possible_keys = [
                        behavior_name,
                        self.normalize_behavior_name(behavior_name),
                        behavior_name.replace('_', ' '),
                        f"behavior_{behavior_name}",
                        f"behavior_{self.normalize_behavior_name(behavior_name)}"
                    ]
                    
                    for key in possible_keys:
                        if key in indicators:
                            raw_value = indicators[key]
                            value = self._convert_behavior_value(raw_value)
                            if value > 0:
                                found_behaviors += 1
                                behavior_type_counts[behavior_name] += 1
                            break
                
                # Also check for direct keys in data
                if value == 0.0:
                    for key in [f"behavior_{behavior_name}", 
                               f"behavior_{self.normalize_behavior_name(behavior_name)}"]:
                        if key in data:
                            raw_value = data[key]
                            value = self._convert_behavior_value(raw_value)
                            if value > 0:
                                found_behaviors += 1
                                behavior_type_counts[behavior_name] += 1
                            break
                
                behavior_values.append(value)
            
            # Store behavior data
            behavior_data[img_id] = behavior_values
            behavior_data[os.path.basename(img_id)] = behavior_values  # Also store by basename
            
            if found_behaviors > 0:
                annotations_with_behaviors += 1
                total_behavior_instances += found_behaviors
        
        # Print statistics
        logger.info(f"Behavioral data extraction results:")
        logger.info(f"  - Total annotations: {len(annotations)}")
        logger.info(f"  - Annotations with behaviors: {annotations_with_behaviors}")
        logger.info(f"  - Total behavior instances: {total_behavior_instances}")
        logger.info(f"  - Average behaviors per annotation: {total_behavior_instances / len(annotations):.2f}")
        
        if debug and behavior_type_counts:
            logger.info("Top 10 most common behaviors:")
            sorted_behaviors = sorted(behavior_type_counts.items(), key=lambda x: x[1], reverse=True)
            for behavior, count in sorted_behaviors[:10]:
                logger.info(f"  - {behavior}: {count} instances")
        
        return behavior_data, self.behavior_columns
    
    def _convert_behavior_value(self, value):
        """
        Convert behavior value to numeric 0/1
        """
        if pd.isna(value) or value == '' or value is None:
            return 0.0
        
        # Convert to string for comparison
        str_val = str(value).lower().strip()
        
        # True values
        if str_val in ['true', '1', 'yes', 'present', 'observed']:
            return 1.0
        
        # False values
        if str_val in ['false', '0', 'no', 'absent', 'not_observed']:
            return 0.0
        
        # Try numeric conversion
        try:
            float_val = float(str_val)
            return 1.0 if float_val > 0 else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def load_annotations(self, split_name='train'):
        """
        Enhanced annotation loading with debugging
        """
        split_dir = os.path.join(self.processed_dir, split_name)
        if not os.path.exists(split_dir):
            logger.error(f"Split directory not found: {split_dir}")
            return None
        
        # Try JSON first
        json_paths = [
            os.path.join(split_dir, "annotations", "annotations.json"),
            os.path.join(split_dir, "annotations.json"),
            os.path.join(split_dir, "annotations", "annotations_fixed.json"),
            os.path.join(split_dir, "annotations_fixed.json"),
        ]
        
        for json_path in json_paths:
            if os.path.exists(json_path):
                logger.info(f"Loading annotations from {json_path}")
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        annotations = json.load(f)
                    logger.info(f"Successfully loaded {len(annotations)} annotations")
                    return annotations
                except Exception as e:
                    logger.warning(f"Error loading {json_path}: {e}")
        
        # Try CSV as fallback
        csv_paths = [
            os.path.join(split_dir, "annotations.csv"),
            os.path.join(split_dir, "annotations_fixed.csv"),
        ]
        
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                logger.info(f"Loading annotations from CSV: {csv_path}")
                try:
                    df = pd.read_csv(csv_path)
                    logger.info(f"Loaded {len(df)} rows from CSV")
                    
                    # Convert to dictionary format
                    annotations = {}
                    behavior_cols = [col for col in df.columns if col.startswith('behavior_')]
                    
                    for idx, row in df.iterrows():
                        img_id = row.get('image_path', f'image_{idx}')
                        
                        annotation = {
                            'emotions': {'primary_emotion': row.get('primary_emotion', 'Unknown')},
                            'source': row.get('source', 'unknown'),
                            'behavioral_indicators': {}
                        }
                        
                        # Extract behavioral indicators from CSV
                        for col in behavior_cols:
                            if pd.notna(row[col]):
                                behavior_name = col.replace('behavior_', '')
                                annotation['behavioral_indicators'][behavior_name] = self._convert_behavior_value(row[col])
                        
                        annotations[img_id] = annotation
                    
                    logger.info(f"Converted CSV to {len(annotations)} annotations")
                    return annotations
                    
                except Exception as e:
                    logger.warning(f"Error loading CSV {csv_path}: {e}")
        
        logger.error(f"No valid annotation files found for {split_name}")
        return None
    
    def diagnose_behavioral_data(self, split_name='train'):
        """
        Comprehensive diagnosis of behavioral data issues
        """
        logger.info("=" * 60)
        logger.info(f"BEHAVIORAL DATA DIAGNOSIS - {split_name.upper()} SPLIT")
        logger.info("=" * 60)
        
        # Load annotations
        annotations = self.load_annotations(split_name)
        if not annotations:
            logger.error("Cannot diagnose - no annotations loaded")
            return
        
        # Load behavior matrix
        matrix_loaded = self.load_behavior_matrix_enhanced(debug=True)
        
        # Analyze annotations structure
        logger.info(f"\n1. ANNOTATION STRUCTURE ANALYSIS:")
        logger.info(f"   - Total annotations: {len(annotations)}")
        
        # Check for behavioral indicators
        with_behavioral_indicators = 0
        without_behavioral_indicators = 0
        total_behaviors = 0
        
        for img_id, data in annotations.items():
            if 'behavioral_indicators' in data and data['behavioral_indicators']:
                with_behavioral_indicators += 1
                total_behaviors += len(data['behavioral_indicators'])
            else:
                without_behavioral_indicators += 1
        
        logger.info(f"   - With behavioral indicators: {with_behavioral_indicators}")
        logger.info(f"   - Without behavioral indicators: {without_behavioral_indicators}")
        logger.info(f"   - Total behavior instances: {total_behaviors}")
        logger.info(f"   - Average behaviors per annotation: {total_behaviors / len(annotations):.2f}")
        
        # Check emotion distribution
        emotion_counts = defaultdict(int)
        for data in annotations.values():
            if 'emotions' in data and 'primary_emotion' in data['emotions']:
                emotion = data['emotions']['primary_emotion']
                emotion_counts[emotion] += 1
        
        logger.info(f"\n2. EMOTION DISTRIBUTION:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percent = count / len(annotations) * 100
            logger.info(f"   - {emotion}: {count} ({percent:.1f}%)")
        
        # Extract and analyze behavioral data
        logger.info(f"\n3. BEHAVIORAL DATA EXTRACTION:")
        behavior_data, behavior_columns = self.load_behavior_data_enhanced(annotations, debug=True)
        
        # Check matrix integration
        logger.info(f"\n4. BEHAVIOR MATRIX INTEGRATION:")
        logger.info(f"   - Matrix loaded: {matrix_loaded}")
        logger.info(f"   - Behavior columns from matrix: {len(self.behavior_columns)}")
        logger.info(f"   - Behaviors extracted from data: {len(behavior_data)}")
        
        if matrix_loaded and self.behavior_matrix:
            matrix_behaviors = list(self.behavior_matrix.get('behaviors', {}).keys())
            logger.info(f"   - Matrix behavior names: {matrix_behaviors[:5]}...")
        
        logger.info("=" * 60)
        
        return {
            'annotations': annotations,
            'behavior_data': behavior_data,
            'behavior_columns': behavior_columns,
            'matrix_loaded': matrix_loaded,
            'statistics': {
                'total_annotations': len(annotations),
                'with_behaviors': with_behavioral_indicators,
                'without_behaviors': without_behavioral_indicators,
                'total_behavior_instances': total_behaviors,
                'emotion_distribution': dict(emotion_counts)
            }
        }

# Usage example
def run_diagnosis(base_dir="C:/Users/kelly/Documents/GitHub/Pawnder"):
    """
    Run comprehensive behavioral data diagnosis
    """
    classifier = FixedDogEmotionWithBehaviors(base_dir)
    
    # Diagnose each split
    for split in ['train', 'validation', 'test']:
        results = classifier.diagnose_behavioral_data(split)
        
        if results:
            print(f"\n{split.upper()} SPLIT SUMMARY:")
            stats = results['statistics']
            print(f"  Total annotations: {stats['total_annotations']}")
            print(f"  With behaviors: {stats['with_behaviors']} ({stats['with_behaviors']/stats['total_annotations']*100:.1f}%)")
            print(f"  Average behaviors per annotation: {stats['total_behavior_instances']/stats['total_annotations']:.2f}")

if __name__ == "__main__":
    run_diagnosis()
