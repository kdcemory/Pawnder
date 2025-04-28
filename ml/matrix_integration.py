"""
Primary Behavior Matrix Integration Script

This script processes the Primary Behavior Matrix into a format usable by the 
dog emotion model and integrates behavioral features with image annotations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from tqdm import tqdm

class BehaviorMatrixIntegrator:
    """
    Class for integrating the Primary Behavior Matrix with dog emotion data
    """
    
    def __init__(self):
        """Initialize with paths"""
        # Fixed paths based on your directory structure
        self.base_dir = "/content/drive/MyDrive/Colab Notebooks/Pawnder"
        self.processed_dir = os.path.join(self.base_dir, "Data/processed")
        self.matrix_dir = os.path.join(self.base_dir, "Data/Matrix")
        
        # Emotion mapping from matrix to class names
        self.emotion_mapping = {
            "Happy/Playful": "Happy",
            "Relaxed": "Relaxed",
            "Submissive/Appeasement": "Submissive",
            "Curiosity/Alertness": "Curiosity",
            "Stressed": "Stressed",
            "Fearful/Anxious": "Fearful",
            "Aggressive/Threatening": "Aggressive"
        }
        
        # Matrix data
        self.matrix_data = None
        self.behavior_columns = []
        self.matrix_behaviors = {}
        
        # Annotations data
        self.annotations = {}
        
        # Print initialization info
        print(f"Initialized Matrix Integrator")
        print(f"Base directory: {self.base_dir}")
        print(f"Matrix directory: {self.matrix_dir}")
        print(f"Processed directory: {self.processed_dir}")
    
    def load_matrix(self):
        """Load Primary Behavior Matrix from file"""
        print("\nLoading Primary Behavior Matrix...")
        
        # Try different possible file locations
        matrix_paths = [
            os.path.join(self.matrix_dir, "Primary Behavior Matrix.xlsx"),
            os.path.join(self.matrix_dir, "primary_behavior_matrix.json"),
            os.path.join(self.base_dir, "Data/Matrix/Primary Behavior Matrix.xlsx"),
            os.path.join(self.base_dir, "Data/Matrix/primary_behavior_matrix.json")
        ]
        
        for matrix_path in matrix_paths:
            if os.path.exists(matrix_path):
                print(f"Found matrix file: {matrix_path}")
                
                try:
                    # Handle Excel format
                    if matrix_path.endswith(".xlsx"):
                        # Load Excel file
                        matrix_df = pd.read_excel(matrix_path, sheet_name=0)
                        print(f"Loaded Excel matrix with {len(matrix_df)} rows and {len(matrix_df.columns)} columns")
                        
                        # Process the matrix based on its structure
                        return self._process_excel_matrix(matrix_df)
                    
                    # Handle JSON format
                    elif matrix_path.endswith(".json"):
                        with open(matrix_path, 'r') as f:
                            matrix_data = json.load(f)
                        
                        print(f"Loaded JSON matrix with {len(matrix_data)} entries")
                        return self._process_json_matrix(matrix_data)
                
                except Exception as e:
                    print(f"Error processing matrix file: {str(e)}")
        
        # If we get here, we couldn't load the matrix
        print("Failed to load Primary Behavior Matrix from any location")
        
        # Create default behavior features
        print("Creating default behavior features based on known categories")
        self._create_default_behavior_features()
        
        return False
    
    def _process_excel_matrix(self, matrix_df):
        """Process Excel format of Primary Behavior Matrix"""
        try:
            # Check the structure of the matrix
            if len(matrix_df) < 5:
                print("Matrix seems too small, may not be correctly formatted")
                return False
            
            # Extract behavior categories
            categories = []
            current_category = None
            
            # Extract behavior columns based on header row and behavior states
            behavior_states = ["Happy/Playful", "Relaxed", "Submissive/Appeasement", 
                              "Curiosity/Alertness", "Stressed", "Fearful/Anxious", 
                              "Aggressive/Threatening"]
            
            # Map header rows to emotion states
            header_mapping = {}
            headers = matrix_df.columns.tolist()
            
            # First, try to find header row with emotion states
            for i, header in enumerate(headers):
                if isinstance(header, str) and any(state in header for state in behavior_states):
                    for state in behavior_states:
                        if state in header:
                            header_mapping[i] = state
            
            # If no columns are found, try different approaches
            if not header_mapping:
                # Check for numeric columns that might contain binary indicators
                numeric_cols = matrix_df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= len(behavior_states):
                    # Assume columns correspond to states in order
                    for i, state in enumerate(behavior_states):
                        if i < len(numeric_cols):
                            header_mapping[matrix_df.columns.get_loc(numeric_cols[i])] = state
            
            # Process rows to extract behaviors
            behaviors = {}
            
            for _, row in matrix_df.iterrows():
                if pd.isna(row.iloc[0]) or row.iloc[0] == '':
                    continue
                
                # Check if this is a category header
                first_val = str(row.iloc[0])
                if all(pd.isna(val) or val == '' or val == 0 for val in row.iloc[1:]):
                    current_category = first_val
                    categories.append(current_category)
                    continue
                
                # This is a behavior under the current category
                behavior_name = row.iloc[0]
                if current_category:
                    behavior_name = f"{current_category} - {behavior_name}"
                
                # Convert to safe column name
                column_name = f"behavior_{behavior_name.lower().replace(' ', '_').replace('/', '_')}"
                
                # Extract behavior indicators
                behavior_indicators = {}
                for col_idx, state in header_mapping.items():
                    if col_idx < len(row):
                        value = row.iloc[col_idx]
                        if not pd.isna(value) and value != '' and value != 0:
                            behavior_indicators[state] = 1
                        else:
                            behavior_indicators[state] = 0
                
                behaviors[column_name] = behavior_indicators
            
            # Store behavior columns and matrix data
            self.behavior_columns = list(behaviors.keys())
            self.matrix_behaviors = behaviors
            
            print(f"Processed {len(self.behavior_columns)} behavior features from Primary Behavior Matrix")
            return True
        
        except Exception as e:
            print(f"Error processing Excel matrix: {str(e)}")
            return False
    
    def _process_json_matrix(self, matrix_data):
        """Process JSON format of Primary Behavior Matrix"""
        try:
            # Extract behavior columns
            behavior_columns = []
            
            # Check if matrix_data is a dict with behavior columns
            if isinstance(matrix_data, dict):
                # Look for behavior_* keys
                behavior_columns = [k for k in matrix_data.keys() 
                                  if isinstance(k, str) and k.startswith('behavior_')]
                
                if behavior_columns:
                    self.behavior_columns = behavior_columns
                    self.matrix_data = matrix_data
                    
                    # Extract behavior indicators
                    behaviors = {}
                    
                    for col in behavior_columns:
                        behaviors[col] = {}
                        
                        # Check if the column has emotion mappings
                        if isinstance(matrix_data[col], dict):
                            for emotion, value in matrix_data[col].items():
                                behaviors[col][emotion] = 1 if value else 0
                    
                    self.matrix_behaviors = behaviors
                    
                    print(f"Processed {len(self.behavior_columns)} behavior features from JSON matrix")
                    return True
            
            print("JSON matrix format not recognized")
            return False
        
        except Exception as e:
            print(f"Error processing JSON matrix: {str(e)}")
            return False
    
    def _create_default_behavior_features(self):
        """Create default behavior features based on known categories"""
        behavior_states = ["Happy/Playful", "Relaxed", "Submissive/Appeasement", 
                          "Curiosity/Alertness", "Stressed", "Fearful/Anxious", 
                          "Aggressive/Threatening"]
        
        # Create default behavior columns
        default_columns = [
            "behavior_tail_high", "behavior_tail_low", "behavior_tail_wagging",
            "behavior_ears_forward", "behavior_ears_back", "behavior_ears_relaxed",
            "behavior_teeth_showing", "behavior_mouth_open", "behavior_mouth_closed",
            "behavior_eyes_wide", "behavior_eyes_squinting", "behavior_pupils_dilated",
            "behavior_posture_tall", "behavior_posture_low", "behavior_posture_stiff"
        ]
        
        # Create default mappings
        behaviors = {}
        
        # Map behaviors to states based on knowledge from the Primary Behavior Matrix
        behavior_mappings = {
            "behavior_tail_high": ["Happy/Playful", "Curiosity/Alertness", "Aggressive/Threatening"],
            "behavior_tail_low": ["Submissive/Appeasement", "Fearful/Anxious"],
            "behavior_tail_wagging": ["Happy/Playful", "Stressed"],
            "behavior_ears_forward": ["Happy/Playful", "Curiosity/Alertness", "Aggressive/Threatening"],
            "behavior_ears_back": ["Submissive/Appeasement", "Fearful/Anxious"],
            "behavior_ears_relaxed": ["Relaxed"],
            "behavior_teeth_showing": ["Aggressive/Threatening", "Fearful/Anxious"],
            "behavior_mouth_open": ["Happy/Playful", "Relaxed"],
            "behavior_mouth_closed": ["Submissive/Appeasement", "Fearful/Anxious"],
            "behavior_eyes_wide": ["Fearful/Anxious", "Curiosity/Alertness"],
            "behavior_eyes_squinting": ["Relaxed", "Submissive/Appeasement"],
            "behavior_pupils_dilated": ["Fearful/Anxious", "Aggressive/Threatening"],
            "behavior_posture_tall": ["Happy/Playful", "Aggressive/Threatening"],
            "behavior_posture_low": ["Submissive/Appeasement", "Fearful/Anxious"],
            "behavior_posture_stiff": ["Stressed", "Aggressive/Threatening"]
        }
        
        # Create behavior indicators
        for column in default_columns:
            behaviors[column] = {}
            
            for state in behavior_states:
                if column in behavior_mappings and state in behavior_mappings[column]:
                    behaviors[column][state] = 1
                else:
                    behaviors[column][state] = 0
        
        # Store behavior columns and mappings
        self.behavior_columns = default_columns
        self.matrix_behaviors = behaviors
        
        print(f"Created {len(self.behavior_columns)} default behavior features")
        return True
    
    def load_annotations(self):
        """Load image annotations from processed directory"""
        print("\nLoading image annotations...")
        
        # Try different annotation file locations
        annotation_files = [
            os.path.join(self.processed_dir, "combined_annotations.json"),
            os.path.join(self.processed_dir, "annotations.json"),
            os.path.join(self.base_dir, "Data/processed/combined_annotations.json"),
            os.path.join(self.base_dir, "Data/processed/annotations.json")
        ]
        
        for annotation_path in annotation_files:
            if os.path.exists(annotation_path):
                print(f"Found annotations file: {annotation_path}")
                
                try:
                    with open(annotation_path, 'r') as f:
                        annotations = json.load(f)
                    
                    print(f"Loaded {len(annotations)} annotations")
                    self.annotations = annotations
                    
                    # Check if annotations already have behavior features
                    has_behaviors = False
                    if annotations:
                        first_key = next(iter(annotations))
                        first_entry = annotations[first_key]
                        
                        # Check for behavior_ keys
                        behavior_keys = [k for k in first_entry.keys() 
                                      if isinstance(k, str) and k.startswith('behavior_')]
                        
                        if behavior_keys:
                            has_behaviors = True
                            print(f"Annotations already contain {len(behavior_keys)} behavior features")
                    
                    return has_behaviors
                
                except Exception as e:
                    print(f"Error loading annotations: {str(e)}")
        
        print("Failed to load annotations from any location")
        return False
    
    def integrate_behaviors(self):
        """Integrate behavior features into annotations"""
        if not self.annotations:
            print("No annotations loaded. Please load annotations first.")
            return False
        
        if not self.behavior_columns:
            print("No behavior features available. Please load matrix or create default features.")
            return False
        
        print("\nIntegrating behavior features into annotations...")
        
        # Check if annotations already have behavior features
        if any(key.startswith('behavior_') for key in next(iter(self.annotations.values()))):
            choice = input("Annotations already have behavior features. Overwrite? (y/n): ")
            if choice.lower() != 'y':
                print("Keeping existing behavior features")
                return True
        
        # Count how many items we'll need to process
        total_items = len(self.annotations)
        
        # Create default behavior values array (all zeros)
        default_behaviors = {col: 0 for col in self.behavior_columns}
        
        # Count how many items have emotions assigned
        emotion_count = 0
        behavior_match_count = 0
        
        # For each annotation, add behavior features based on emotion
        for img_id, data in tqdm(self.annotations.items(), total=total_items, desc="Adding behaviors"):
            # Copy default behaviors
            behaviors = default_behaviors.copy()
            
            # Check if this item has an emotion assigned
            if "emotions" in data and "primary_emotion" in data["emotions"]:
                emotion = data["emotions"]["primary_emotion"]
                emotion_count += 1
                
                # For each behavior column, check if it should be active for this emotion
                for col, emotion_map in self.matrix_behaviors.items():
                    if col in self.behavior_columns:
                        if emotion in emotion_map and emotion_map[emotion] == 1:
                            behaviors[col] = 1
                            behavior_match_count += 1
            
            # Add behavior features to annotation
            self.annotations[img_id].update(behaviors)
        
        print(f"Added behavior features to {total_items} annotations")
        print(f"  - {emotion_count} annotations had emotion data")
        print(f"  - {behavior_match_count} behavior features were activated based on emotions")
        
        return True
    
    def save_integrated_annotations(self):
        """Save integrated annotations back to file"""
        if not self.annotations:
            print("No annotations to save")
            return False
        
        print("\nSaving integrated annotations...")
        
        # Create output file path
        output_dir = os.path.join(self.processed_dir, "with_behaviors")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(output_dir, f"annotations_with_behaviors_{timestamp}.json")
        
        # Save annotations
        with open(output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print(f"Saved integrated annotations to {output_file}")
        
        # Also save to standard location for the model
        standard_path = os.path.join(self.processed_dir, "combined_annotations.json")
        with open(standard_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print(f"Also saved to standard location: {standard_path}")
        
        return output_file
    
    def analyze_behavior_features(self):
        """Analyze and visualize behavior features"""
        if not self.annotations:
            print("No annotations to analyze")
            return
        
        if not self.behavior_columns:
            print("No behavior features to analyze")
            return
        
        print("\nAnalyzing behavior features...")
        
        # Count behavior feature occurrence by emotion
        behavior_by_emotion = defaultdict(lambda: defaultdict(int))
        emotion_counts = defaultdict(int)
        
        # For each annotation, count behavior features by emotion
        for img_id, data in self.annotations.items():
            # Check if this item has an emotion assigned
            if "emotions" in data and "primary_emotion" in data["emotions"]:
                emotion = data["emotions"]["primary_emotion"]
                emotion_counts[emotion] += 1
                
                # Count active behavior features
                for col in self.behavior_columns:
                    if col in data and data[col] == 1:
                        behavior_by_emotion[emotion][col] += 1
        
        # Print summary
        print("\nBehavior Feature Summary by Emotion:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"\n{emotion} ({count} annotations):")
            
            # Get behavior features for this emotion
            features = behavior_by_emotion[emotion]
            
            if not features:
                print("  No active behavior features")
                continue
            
            # Sort by frequency
            for feature, feature_count in sorted(features.items(), key=lambda x: x[1], reverse=True):
                percentage = feature_count / count * 100
                print(f"  {feature}: {feature_count} ({percentage:.1f}%)")
        
        # Create visualization of behavior features
        print("\nCreating behavior feature visualization...")
        
        try:
            # Create matrix of behavior features by emotion
            emotions = list(emotion_counts.keys())
            features = self.behavior_columns
            
            # Normalize by number of annotations for each emotion
            matrix = np.zeros((len(emotions), len(features)))
            
            for i, emotion in enumerate(emotions):
                emotion_count = emotion_counts[emotion]
                
                if emotion_count > 0:
                    for j, feature in enumerate(features):
                        matrix[i, j] = behavior_by_emotion[emotion].get(feature, 0) / emotion_count
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            plt.imshow(matrix, cmap='viridis', aspect='auto')
            
            # Add labels
            plt.yticks(range(len(emotions)), emotions)
            plt.xticks(range(len(features)), [f.replace('behavior_', '') for f in features], rotation=90)
            
            plt.colorbar(label='Proportion of annotations')
            plt.title('Behavior Features by Emotion')
            plt.tight_layout()
            
            # Save visualization
            output_dir = os.path.join(self.processed_dir, "visualizations")
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, "behavior_features_by_emotion.png")
            plt.savefig(output_file, dpi=300)
            
            print(f"Visualization saved to {output_file}")
            
            # Show in notebook if running in one
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
    
    def run_integration_pipeline(self):
        """Run the complete integration pipeline"""
        print("\n" + "="*80)
        print("Primary Behavior Matrix Integration Pipeline")
        print("="*80)
        
        # Step 1: Load the behavior matrix
        matrix_loaded = self.load_matrix()
        
        # Step 2: Load annotations
        has_behaviors = self.load_annotations()
        
        # Step 3: Integrate behaviors if needed
        if not has_behaviors:
            self.integrate_behaviors()
        else:
            choice = input("Annotations already have behavior features. Reintegrate? (y/n): ")
            if choice.lower() == 'y':
                self.integrate_behaviors()
        
        # Step 4: Save integrated annotations
        self.save_integrated_annotations()
        
        # Step 5: Analyze behavior features
        self.analyze_behavior_features()
        
        print("\n" + "="*80)
        print("Integration pipeline completed!")
        print("="*80)


# Run the integration if executed directly
if __name__ == "__main__":
    integrator = BehaviorMatrixIntegrator()
    integrator.run_integration_pipeline()