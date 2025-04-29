"""
Improved Dog Emotion Classifier with Behavioral Feature Integration

This implementation properly handles behavioral indicators from annotations,
fixes path issues, and supports both image-based and behavioral feature-based analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import re
import glob

# Try to configure GPU settings
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s). Memory growth enabled")
        
    # Try to enable mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")
except Exception as e:
    print(f"GPU configuration error: {str(e)}")

def find_directory(base_dir="/content/drive/MyDrive/Colab Notebooks/Pawnder", target_dir=None):
    """
    Find directories in the project structure to help with auto-detection
    
    Args:
        base_dir: Directory to start the search
        target_dir: Optional specific directory to find
    
    Returns:
        Dictionary with paths to key directories
    """
    paths = {}
    
    # Try to locate key directories
    if target_dir:
        # Search for a specific directory
        for root, dirs, files in os.walk(base_dir):
            if target_dir in dirs:
                paths[target_dir] = os.path.join(root, target_dir)
                return paths
    else:
        # Look for main project directories
        for root, dirs, files in os.walk(base_dir):
            # Look for train_by_class directory which indicates processed data
            if 'train_by_class' in dirs:
                paths['train_by_class'] = os.path.join(root, 'train_by_class')
                paths['processed_dir'] = root
                
            # Look for Matrix directory which contains the behavior matrix
            if 'Matrix' in dirs:
                paths['matrix_dir'] = os.path.join(root, 'Matrix')
                
            # Look for Models directory
            if 'Models' in dirs:
                paths['model_dir'] = os.path.join(root, 'Models')
    
    # Debug what was found
    if paths:
        print("Found project directories:")
        for key, path in paths.items():
            print(f"  {key}: {path}")
    else:
        print(f"Could not find project directories in {base_dir}")
        
    return paths

class DogEmotionWithBehaviors:
    """
    Dog emotion classification model with behavioral feature integration
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize the classifier with auto-detected paths
        
        Args:
            base_dir: Optional base directory, if None will try to auto-detect
        """
        # Auto-detect project structure if base_dir not provided
        if base_dir is None:
            # First try the default location
            default_dir = "/content/drive/MyDrive/Colab Notebooks/Pawnder"
            if os.path.exists(default_dir):
                base_dir = default_dir
            else:
                # Look for potential matches on Google Drive
                drive_dir = "/content/drive/MyDrive"
                if os.path.exists(drive_dir):
                    potential_dirs = [
                        os.path.join(drive_dir, d) for d in os.listdir(drive_dir)
                        if 'pawnder' in d.lower() or 'dog' in d.lower()
                    ]
                    if potential_dirs:
                        base_dir = potential_dirs[0]
                        print(f"Auto-detected project directory: {base_dir}")
        
        # Store base directory
        self.base_dir = base_dir or "/content/drive/MyDrive/Colab Notebooks/Pawnder"
        
        # Auto-detect other directories
        paths = find_directory(self.base_dir)
        
        # Set up paths based on auto-detection or defaults
        self.processed_dir = paths.get('processed_dir', os.path.join(self.base_dir, "Data/processed"))
        self.matrix_dir = paths.get('matrix_dir', os.path.join(self.base_dir, "Data/Matrix"))
        self.model_dir = paths.get('model_dir', os.path.join(self.base_dir, "Models"))
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load behavior matrix if available
        self.matrix_data = self.load_behavior_matrix()
        
        # Standard emotion class names
        self.standard_emotions = [
            "Happy/Playful", 
            "Relaxed", 
            "Submissive/Appeasement", 
            "Curiosity/Alertness", 
            "Stressed", 
            "Fearful/Anxious", 
            "Aggressive/Threatening"
        ]
        
        # Safe class names (for directory names)
        self.safe_class_names = [
            "Happy_Playful", 
            "Relaxed", 
            "Submissive_Appeasement", 
            "Curiosity_Alertness", 
            "Stressed", 
            "Fearful_Anxious", 
            "Aggressive_Threatening"
        ]
        
        # Map between different naming conventions
        self.class_name_mapping = {
            # From directory naming to standard naming
            "Happy": "Happy/Playful",
            "Relaxed": "Relaxed",
            "Submissive": "Submissive/Appeasement",
            "Curiosity": "Curiosity/Alertness",
            "Stressed": "Stressed",
            "Fearful": "Fearful/Anxious",
            "Aggressive": "Aggressive/Threatening",
            
            # From standard naming to safe naming
            "Happy/Playful": "Happy_Playful",
            "Relaxed": "Relaxed",
            "Submissive/Appeasement": "Submissive_Appeasement",
            "Curiosity/Alertness": "Curiosity_Alertness",
            "Stressed": "Stressed",
            "Fearful/Anxious": "Fearful_Anxious",
            "Aggressive/Threatening": "Aggressive_Threatening",
        }
        
        # Reverse mapping (safe to standard)
        self.safe_to_standard = {
            "Happy_Playful": "Happy/Playful",
            "Relaxed": "Relaxed",
            "Submissive_Appeasement": "Submissive/Appeasement",
            "Curiosity_Alertness": "Curiosity/Alertness",
            "Stressed": "Stressed",
            "Fearful_Anxious": "Fearful/Anxious",
            "Aggressive_Threatening": "Aggressive/Threatening",
        }
        
        self.model = None
        self.class_names = []
        self.behavior_columns = []
        
        # Print configuration for verification
        print(f"Base directory: {self.base_dir}")
        print(f"Processed directory: {self.processed_dir}")
        print(f"Matrix directory: {self.matrix_dir}")
        print(f"Model directory: {self.model_dir}")
        
        # Verify critical directories
        print("\nChecking directory structure:")
        
        # Check class-based directories
        for split_name, dir_name in [
            ("Train (class-based)", "train_by_class"),
            ("Validation (class-based)", "val_by_class"),
            ("Test (class-based)", "test_by_class")
        ]:
            split_dir = os.path.join(self.processed_dir, dir_name)
            exists = os.path.exists(split_dir)
            print(f"  {split_name} directory exists: {exists}")
            if exists:
                class_dirs = [d for d in os.listdir(split_dir) 
                             if os.path.isdir(os.path.join(split_dir, d))]
                print(f"    Contains classes: {', '.join(class_dirs[:5])}{' and more...' if len(class_dirs) > 5 else ''}")
        
        # Check direct split directories with annotations
        for split_name, dir_name in [
            ("Train (direct)", "train"),
            ("Validation (direct)", "validation"),
            ("Test (direct)", "test")
        ]:
            split_dir = os.path.join(self.processed_dir, dir_name)
            exists = os.path.exists(split_dir)
            
            # Check for JSON annotations
            json_path = os.path.join(split_dir, "annotations", "annotations.json")
            json_exists = os.path.exists(json_path)
            
            # Check for CSV annotations
            csv_path = os.path.join(split_dir, "annotations.csv")
            csv_exists = os.path.exists(csv_path)
            
            print(f"  {split_name} directory exists: {exists}")
            if exists:
                print(f"    Has JSON annotations: {json_exists}")
                print(f"    Has CSV annotations: {csv_exists}")
                
        # Print behavior matrix info
        if self.matrix_data:
            print(f"\nBehavior matrix loaded with {len(self.get_behavior_columns())} behavioral features")
        else:
            print("\nNo behavior matrix found, will use default behavioral features")
    
    def load_behavior_matrix(self):
        """
        Load the Primary Behavior Matrix data
        
        Returns:
            dict: Matrix data or None if not found
        """
        # Try different possible file locations
        matrix_paths = [
            os.path.join(self.matrix_dir, "primary_behavior_matrix.json"),
            os.path.join(self.matrix_dir, "Primary Behavior Matrix.xlsx"),
            # Check other potential locations
            os.path.join(self.base_dir, "Data/Matrix/primary_behavior_matrix.json"),
            os.path.join(self.base_dir, "Data/Matrix/Primary Behavior Matrix.xlsx"),
        ]
        
        # Try to find and load the matrix
        for matrix_path in matrix_paths:
            if os.path.exists(matrix_path):
                print(f"Found behavior matrix at {matrix_path}")
                
                try:
                    # Handle JSON format
                    if matrix_path.endswith('.json'):
                        with open(matrix_path, 'r') as f:
                            return json.load(f)
                    
                    # Handle Excel format
                    elif matrix_path.endswith('.xlsx'):
                        import pandas as pd
                        # Read Excel file
                        excel_df = pd.read_excel(matrix_path)
                        
                        # Process the matrix based on its structure (implementation depends on format)
                        # For now, return the DataFrame as is
                        return self._process_excel_matrix(excel_df)
                
                except Exception as e:
                    print(f"Error loading behavior matrix: {str(e)}")
        
        print("No behavior matrix found")
        return None
    
    def _process_excel_matrix(self, matrix_df):
        """
        Process the Excel format of the Primary Behavior Matrix
        
        Args:
            matrix_df: DataFrame containing the matrix
            
        Returns:
            dict: Processed matrix data
        """
        # Extract the behavior categories and indicators
        behavior_data = {"categories": [], "behaviors": {}, "mappings": {}}
        
        # This implementation depends on the exact structure of your Excel file
        # Here's a generic implementation based on the structure described
        try:
            # Extract header information (assuming it's in the first few rows)
            headers = matrix_df.iloc[0:5].values
            
            # For each following row, extract behavior information
            current_category = None
            
            for idx, row in matrix_df.iloc[5:].iterrows():
                row_data = row.values
                
                # Check if this is a category header (typically has no values in other columns)
                if pd.notna(row_data[0]) and all(pd.isna(val) or val == 0 or val == "" for val in row_data[1:]):
                    current_category = str(row_data[0])
                    behavior_data["categories"].append(current_category)
                    continue
                
                # If this is a behavior row
                if pd.notna(row_data[0]) and current_category is not None:
                    behavior_name = str(row_data[0])
                    behavior_id = f"behavior_{current_category.lower().replace(' ', '_').replace('/', '_')}_{behavior_name.lower().replace(' ', '_').replace('/', '_')}"
                    
                    # Create mapping to emotional states
                    mapping = {}
                    for col_idx, col_name in enumerate(matrix_df.columns[1:8], 1):  # Assuming columns B-H are behavioral states
                        value = row_data[col_idx]
                        if pd.notna(value) and value != 0 and value != "":
                            # Map column to emotion state (based on row 4 headers)
                            if pd.notna(headers[4, col_idx]):
                                emotion_state = str(headers[4, col_idx])
                                mapping[emotion_state] = 1
                    
                    # Store behavior with its mapping
                    behavior_data["behaviors"][behavior_id] = {
                        "name": behavior_name,
                        "category": current_category
                    }
                    behavior_data["mappings"][behavior_id] = mapping
            
            print(f"Processed Excel matrix with {len(behavior_data['behaviors'])} behaviors in {len(behavior_data['categories'])} categories")
            return behavior_data
        
        except Exception as e:
            print(f"Error processing Excel matrix: {str(e)}")
            return None
    
    def get_behavior_columns(self):
        """
        Get the list of behavior feature columns
        
        Returns:
            list: List of behavior column names
        """
        # If we already processed behavior columns, return them
        if self.behavior_columns:
            return self.behavior_columns
        
        # If we have matrix data, extract columns from it
        if self.matrix_data and "behaviors" in self.matrix_data:
            self.behavior_columns = list(self.matrix_data["behaviors"].keys())
        else:
            # Create default behavior columns
            self.behavior_columns = [
                "behavior_tail_high", "behavior_tail_low", "behavior_tail_wagging",
                "behavior_ears_forward", "behavior_ears_back", "behavior_ears_neutral",
                "behavior_mouth_open", "behavior_mouth_closed", "behavior_teeth_showing",
                "behavior_eyes_wide", "behavior_eyes_squinting", "behavior_posture_tall",
                "behavior_posture_low", "behavior_posture_stiff", "behavior_posture_relaxed"
            ]
        
        return self.behavior_columns
    
    def create_model(self, num_classes, behavior_size=None, img_size=(224, 224, 3)):
        """
        Create model with both image and behavioral inputs
        
        Args:
            num_classes: Number of emotion classes
            behavior_size: Number of behavioral features
            img_size: Input image size (height, width, channels)
            
        Returns:
            Compiled Keras model
        """
        # If behavior_size not provided, use the number of behavior columns
        if behavior_size is None:
            behavior_size = len(self.get_behavior_columns())
        
        # Create image input branch
        image_input = tf.keras.Input(shape=img_size, name='image_input')
        
        # Use MobileNetV2 as base model
        base_model = applications.MobileNetV2(
            input_shape=img_size,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Extract features from image
        x = base_model(image_input)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Create behavioral input branch
        behavior_input = tf.keras.Input(shape=(behavior_size,), name='behavior_input')
        b = layers.Dense(64, activation='relu')(behavior_input)
        b = layers.BatchNormalization()(b)
        b = layers.Dropout(0.3)(b)
        
        # Combine image and behavior branches
        combined = layers.Concatenate()([x, b])
        
        # Classification head
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        
        # Create separate outputs for emotion class and confidence
        emotion_output = layers.Dense(num_classes, activation='softmax', name='emotion_output')(combined)
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence_output')(combined)
        
        # Create and compile model
        model = tf.keras.Model(
            inputs={'image_input': image_input, 'behavior_input': behavior_input},
            outputs=[emotion_output, confidence_output]
        )
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss={
                'emotion_output': 'categorical_crossentropy',
                'confidence_output': 'binary_crossentropy'
            },
            metrics={
                'emotion_output': 'accuracy',
                'confidence_output': 'accuracy'
            },
            loss_weights={
                'emotion_output': 1.0,
                'confidence_output': 0.2
            }
        )
        
        # Store model
        self.model = model
        return model
    
    def fine_tune_model(self, unfreeze_layers=15):
        """
        Fine-tune the model by unfreezing some base model layers
        
        Args:
            unfreeze_layers: Number of layers to unfreeze from the end
            
        Returns:
            Fine-tuned model
        """
        if self.model is None:
            raise ValueError("No model to fine-tune. Create a model first.")
        
        # Find the base model
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break
        
        if base_model is None:
            print("Base model not found. Looking deeper in the model structure...")
            # Try to find it in the functional API structure
            for layer in self.model.layers:
                if hasattr(layer, 'layer') and isinstance(layer.layer, tf.keras.Model):
                    base_model = layer.layer
                    break
        
        if base_model is None:
            print("Could not find base model for fine-tuning")
            return self.model
        
        # Make base model trainable
        base_model.trainable = True
        
        # Freeze all layers except the last few
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss={
                'emotion_output': 'categorical_crossentropy',
                'confidence_output': 'binary_crossentropy'
            },
            metrics={
                'emotion_output': 'accuracy',
                'confidence_output': 'accuracy'
            },
            loss_weights={
                'emotion_output': 1.0,
                'confidence_output': 0.2
            }
        )
        
        print(f"Model fine-tuned: Unfroze last {unfreeze_layers} layers of the base model")
        return self.model
    
    def load_annotations(self, split_name='train'):
        """
        Load annotations from the processed directory for a specific split.
        Handles both JSON and CSV formats.
        
        Args:
            split_name: Data split ('train', 'validation', 'test')
            
        Returns:
            dict: Annotations dictionary or None if error occurs
        """
        # Get split directory
        split_dir = os.path.join(self.processed_dir, split_name)
        if not os.path.exists(split_dir):
            print(f"Split directory not found: {split_dir}")
            # Try alternative names (val vs validation)
            if split_name == 'validation':
                alt_dir = os.path.join(self.processed_dir, 'val')
                if os.path.exists(alt_dir):
                    split_dir = alt_dir
                    print(f"Using alternative directory: {split_dir}")
                else:
                    return None
            else:
                return None
        
        # Define potential annotation paths
        json_paths = [
            os.path.join(split_dir, "annotations", "annotations.json"),
            os.path.join(split_dir, "annotations.json"),
            os.path.join(self.processed_dir, f"{split_name}_annotations.json"),
            os.path.join(self.processed_dir, "annotations", f"{split_name}.json"),
            os.path.join(self.processed_dir, "combined_annotations.json")  # Try the combined annotations
        ]
        
        csv_paths = [
            os.path.join(split_dir, "annotations.csv"),
            os.path.join(split_dir, "annotations", "annotations.csv"),
            os.path.join(self.processed_dir, f"{split_name}_annotations.csv")
        ]
        
        # Try to load JSON annotations first
        annotations = None
        
        # Check each potential JSON path
        for json_path in json_paths:
            if os.path.exists(json_path):
                print(f"Found JSON annotations at {json_path}")
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        all_annotations = json.load(f)
                    
                    # If loading combined annotations, filter for this split
                    if "combined_annotations.json" in json_path:
                        annotations = {}
                        for img_id, data in all_annotations.items():
                            # Check if this annotation belongs to the requested split
                            annotation_split = data.get('split', '').lower()
                            if annotation_split == split_name.lower() or (annotation_split == 'val' and split_name == 'validation'):
                                annotations[img_id] = data
                        
                        if not annotations:
                            # If filtering produced no results, just use all annotations
                            # This is common if the split info is not in the annotations
                            annotations = all_annotations
                            print(f"No split information found in combined annotations, using all {len(annotations)} entries")
                        else:
                            print(f"Filtered combined annotations for {split_name} split: {len(annotations)} entries")
                    else:
                        annotations = all_annotations
                        print(f"Successfully loaded {len(annotations)} JSON annotations")
                    
                    break
                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON file {json_path}: {str(e)}")
                except Exception as e:
                    print(f"Error loading JSON file {json_path}: {str(e)}")
        
        # If JSON loading failed, try CSV
        if annotations is None:
            for csv_path in csv_paths:
                if os.path.exists(csv_path):
                    print(f"Found CSV annotations at {csv_path}")
                    try:
                        # Read CSV using pandas
                        df = pd.read_csv(csv_path)
                        print(f"Successfully loaded {len(df)} CSV rows")
                        
                        # Convert DataFrame to dictionary
                        annotations = {}
                        
                        # Try to determine the ID column - look for columns like 'id', 'image_id', 'filename'
                        id_column = None
                        for col in ['id', 'ID', 'image_id', 'filename', 'image_path', 'file']:
                            if col in df.columns:
                                id_column = col
                                break
                        
                        # If no ID column found, use the index
                        if id_column is None:
                            print("No ID column found in CSV, using DataFrame index")
                            for idx, row in df.iterrows():
                                annotations[f"image_{idx}"] = row.to_dict()
                        else:
                            print(f"Using '{id_column}' as ID column")
                            for idx, row in df.iterrows():
                                annotations[str(row[id_column])] = row.to_dict()
                        
                        # Check if there's an emotions column or primary_emotion column
                        if 'primary_emotion' in df.columns:
                            print("Found 'primary_emotion' column in CSV")
                            # Add emotions structure expected by the model
                            for key in annotations:
                                if 'primary_emotion' in annotations[key]:
                                    primary_emotion = annotations[key]['primary_emotion']
                                    annotations[key]['emotions'] = {'primary_emotion': primary_emotion}
                        
                        break
                    except Exception as e:
                        print(f"Error loading CSV file {csv_path}: {str(e)}")
        
        # If annotations is still None, look for combined annotations and try to filter
        if annotations is None:
            combined_path = os.path.join(self.processed_dir, "combined_annotations.json")
            if os.path.exists(combined_path):
                try:
                    print(f"Trying to load from combined annotations at {combined_path}")
                    with open(combined_path, 'r', encoding='utf-8') as f:
                        all_annotations = json.load(f)
                    
                    # Create split-specific directories to check for image files
                    # This allows us to filter by actually checking which images exist in the split
                    split_images_dir = os.path.join(self.processed_dir, split_name, "images")
                    class_dir = os.path.join(self.processed_dir, f"{split_name}_by_class")
                    
                    if os.path.exists(split_images_dir) or os.path.exists(class_dir):
                        # Filter annotations based on which files exist in this split
                        annotations = {}
                        
                        # Get list of images in this split
                        split_images = set()
                        
                        # Check split/images directory
                        if os.path.exists(split_images_dir):
                            for img in os.listdir(split_images_dir):
                                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    split_images.add(img)
                                    split_images.add(os.path.splitext(img)[0])  # Also add without extension
                        
                        # Check split_by_class directories
                        if os.path.exists(class_dir):
                            for class_subdir in os.listdir(class_dir):
                                subdir_path = os.path.join(class_dir, class_subdir)
                                if os.path.isdir(subdir_path):
                                    for img in os.listdir(subdir_path):
                                        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                                            split_images.add(img)
                                            split_images.add(os.path.splitext(img)[0])
                        
                        print(f"Found {len(split_images)} images in {split_name} split directories")
                        
                        # Filter annotations to match images in this split
                        for img_id, data in all_annotations.items():
                            img_basename = os.path.basename(img_id)
                            img_basename_no_ext = os.path.splitext(img_basename)[0]
                            
                            if img_basename in split_images or img_basename_no_ext in split_images or img_id in split_images:
                                annotations[img_id] = data
                        
                        print(f"Filtered combined annotations to {len(annotations)} entries for {split_name} split")
                    else:
                        # If we can't find the split directories, just use all annotations
                        annotations = all_annotations
                        print(f"Using all {len(annotations)} combined annotations (no split directories found)")
                
                except Exception as e:
                    print(f"Error processing combined annotations: {str(e)}")
        
        # If annotations is still None, we failed to load any annotations
        if annotations is None:
            print(f"Failed to load annotations for {split_name} split")
            return None
        
        # Check annotation format
        if len(annotations) > 0:
            first_key = next(iter(annotations))
            first_item = annotations[first_key]
            
            print(f"Sample annotation key: {first_key}")
            print(f"Sample annotation type: {type(first_item).__name__}")
            
            if isinstance(first_item, dict):
                print(f"Sample annotation fields: {', '.join(list(first_item.keys())[:10])}")
                
                # Check if emotions field exists, and if not, create it
                if 'emotions' not in first_item:
                    print("Adding 'emotions' field to annotations")
                    # Check if there's a primary_emotion field directly in the annotation
                    for key, item in annotations.items():
                        if 'primary_emotion' in item:
                            emotion = item['primary_emotion']
                            item['emotions'] = {'primary_emotion': emotion}
                
                # Extract behavior columns
                behavior_columns = []
                for key, val in first_item.items():
                    if key.startswith('behavior_'):
                        behavior_columns.append(key)
                
                if behavior_columns:
                    print(f"Found {len(behavior_columns)} behavior columns in annotations")
                    self.behavior_columns = behavior_columns
        
        # If no behavior_columns found, use defaults
        if not self.behavior_columns:
            self.behavior_columns = self.get_behavior_columns()
            print(f"Using {len(self.behavior_columns)} default behavior columns")
        
        return annotations
    
    def find_images_for_split(self, split_name='train'):
        """
        Find all image files for a specific split
        
        Args:
            split_name: Data split ('train', 'validation', 'test')
            
        Returns:
            dict: Mapping of image_id to file path
        """
        # Check both split/images and split_by_class directories
        images_dict = {}
        
        # Try split/images directory
        split_images_dir = os.path.join(self.processed_dir, split_name, "images")
        if os.path.exists(split_images_dir):
            for img in os.listdir(split_images_dir):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images_dict[img] = os.path.join(split_images_dir, img)
                    # Also add without extension
                    images_dict[os.path.splitext(img)[0]] = os.path.join(split_images_dir, img)
        
        # Try alternative name for validation
        if split_name == 'validation' and not os.path.exists(split_images_dir):
            val_images_dir = os.path.join(self.processed_dir, 'val', "images")
            if os.path.exists(val_images_dir):
                for img in os.listdir(val_images_dir):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images_dict[img] = os.path.join(val_images_dir, img)
                        # Also add without extension
                        images_dict[os.path.splitext(img)[0]] = os.path.join(val_images_dir, img)
        
        # Try split_by_class directories
        class_dir = os.path.join(self.processed_dir, f"{split_name}_by_class")
        if not os.path.exists(class_dir) and split_name == 'validation':
            # Try alternate name
            class_dir = os.path.join(self.processed_dir, "val_by_class")
        
        if os.path.exists(class_dir):
            for class_subdir in os.listdir(class_dir):
                subdir_path = os.path.join(class_dir, class_subdir)
                if os.path.isdir(subdir_path):
                    for img in os.listdir(subdir_path):
                        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                            images_dict[img] = os.path.join(subdir_path, img)
                            # Also add without extension
                            images_dict[os.path.splitext(img)[0]] = os.path.join(subdir_path, img)
        
        if not images_dict:
            print(f"Warning: No images found for {split_name} split")
            
            # Try looking for any images in all_frames directory
            all_frames_dir = os.path.join(self.processed_dir, "all_frames")
            if os.path.exists(all_frames_dir):
                print(f"Looking for images in {all_frames_dir}")
                for img in os.listdir(all_frames_dir):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images_dict[img] = os.path.join(all_frames_dir, img)
                        # Also add without extension
                        images_dict[os.path.splitext(img)[0]] = os.path.join(all_frames_dir, img)
                
                print(f"Found {len(images_dict)} images in all_frames directory")
        else:
            print(f"Found {len(images_dict)} images for {split_name} split")
            
        return images_dict
    
    def load_data_from_annotations(self, split_name='train'):
        """
        Load image paths and labels from annotations
        
        Args:
            split_name: Data split ('train', 'validation', 'test')
            
        Returns:
            tuple: (image_paths, labels, class_names)
        """
        # Load annotations for this split
        annotations = self.load_annotations(split_name)
        if not annotations:
            raise ValueError(f"Could not load annotations for {split_name} split")
        
        # Find all images for this split
        images_dict = self.find_images_for_split(split_name)
        
        # Collect unique classes
        all_classes = set()
        emotion_counts = {}
        
        for img_id, data in annotations.items():
            # Check if emotions field exists
            if "emotions" in data and "primary_emotion" in data["emotions"]:
                emotion = data["emotions"]["primary_emotion"]
                # Map to safe naming
                safe_emotion = self.class_name_mapping.get(emotion, emotion)
                if '/' in safe_emotion:  # If still has slashes, replace them
                    safe_emotion = safe_emotion.replace('/', '_')
                all_classes.add(safe_emotion)
                emotion_counts[safe_emotion] = emotion_counts.get(safe_emotion, 0) + 1
            # Check if primary_emotion field exists directly
            elif "primary_emotion" in data:
                emotion = data["primary_emotion"]
                safe_emotion = self.class_name_mapping.get(emotion, emotion)
                if '/' in safe_emotion:  # If still has slashes, replace them
                    safe_emotion = safe_emotion.replace('/', '_')
                all_classes.add(safe_emotion)
                emotion_counts[safe_emotion] = emotion_counts.get(safe_emotion, 0) + 1
        
        # If no classes found, use standard classes
        if not all_classes:
            all_classes = set(self.safe_class_names)
            print(f"No classes found in annotations, using standard classes: {all_classes}")
        
        # Sort classes to match standard order where possible
        class_names = []
        # First add standard classes in order
        for cls in self.safe_class_names:
            if cls in all_classes:
                class_names.append(cls)
                all_classes.remove(cls)
        # Then add any remaining non-standard classes
        class_names.extend(sorted(all_classes))
        
        # Print class distribution
        print(f"Class distribution for {split_name} split:")
        for cls, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {count} images ({count/len(annotations)*100:.1f}%)")
        
        # Store class names
        self.class_names = class_names
        
        # Convert to class indices
        class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        
        # Collect image paths and labels
        image_paths = []
        labels = []
        image_ids = []  # Store image IDs for behavioral feature extraction
        
        for img_id, data in annotations.items():
            # Get emotion from either emotions.primary_emotion or primary_emotion
            emotion = None
            if "emotions" in data and "primary_emotion" in data["emotions"]:
                emotion = data["emotions"]["primary_emotion"]
            elif "primary_emotion" in data:
                emotion = data["primary_emotion"]
            
            # Skip if no emotion data
            if emotion is None:
                continue
            
            # Map to safe naming
            safe_emotion = self.class_name_mapping.get(emotion, emotion)
            if '/' in safe_emotion:  # If still has slashes, replace them
                safe_emotion = safe_emotion.replace('/', '_')
            
            # Skip if not in class list
            if safe_emotion not in class_to_idx:
                continue
            
            # Form image path from dictionary of found images
            img_path = None
            
            # Try different keys to match image
            if img_id in images_dict:
                img_path = images_dict[img_id]
            elif os.path.basename(img_id) in images_dict:
                img_path = images_dict[os.path.basename(img_id)]
            elif os.path.splitext(os.path.basename(img_id))[0] in images_dict:
                img_path = images_dict[os.path.splitext(os.path.basename(img_id))[0]]
            
            # If no match found, try a more extensive search
            if img_path is None:
                # Try to find files that contain the image ID
                match_found = False
                for key, path in images_dict.items():
                    # Try to match by various patterns
                    if key == img_id or key.startswith(img_id) or img_id.startswith(key):
                        img_path = path
                        match_found = True
                        break
                
                if not match_found:
                    # Skip this annotation since we couldn't find the image
                    continue
            
            # Check if the image file exists
            if not os.path.exists(img_path):
                continue
            
            # Add the image
            image_paths.append(img_path)
            labels.append(class_to_idx[safe_emotion])
            image_ids.append(img_id)
        
        print(f"Loaded {len(image_paths)} images for {split_name} split")
        
        # Check a few paths
        if image_paths:
            print("Sample image paths:")
            for i in range(min(3, len(image_paths))):
                print(f"  {image_paths[i]}")
                print(f"    Exists: {os.path.exists(image_paths[i])}")
        
        # Return paths, labels, class names, and image IDs for behavior features
        return image_paths, labels, class_names, image_ids
    
    def extract_behavior_features(self, annotations, image_ids):
        """
        Extract behavior features from annotations for specific images
        
        Args:
            annotations: Dictionary of annotations
            image_ids: List of image IDs to extract features for
            
        Returns:
            dict: Dictionary mapping image path to behavior features
        """
        behavior_data = {}
        behavior_columns = self.get_behavior_columns()
        
        print(f"Extracting {len(behavior_columns)} behavior features for {len(image_ids)} images")
        
        # For each image, extract behavior features
        for img_id in image_ids:
            # Check if this image has annotations
            if img_id in annotations:
                # Get annotation data
                data = annotations[img_id]
                
                # Extract behavior features
                features = []
                for col in behavior_columns:
                    # Check if this behavior column exists in the annotation
                    if col in data:
                        value = data[col]
                        # Convert to float
                        if isinstance(value, bool):
                            value = 1.0 if value else 0.0
                        elif isinstance(value, (int, float)):
                            value = float(value)
                        else:
                            # Try to convert to float
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                value = 0.0
                    else:
                        # If behavior not in annotation, check if we can infer it from the emotion
                        value = self._infer_behavior_from_emotion(col, data)
                    
                    features.append(value)
                
                # Store features for this image
                behavior_data[img_id] = features
                
                # Also store by basename for easier matching during training
                basename = os.path.basename(img_id)
                if basename:
                    behavior_data[basename] = features
                
                # Store without extension as well
                basename_no_ext = os.path.splitext(basename)[0]
                if basename_no_ext:
                    behavior_data[basename_no_ext] = features
        
        print(f"Extracted behavior features for {len(behavior_data)} images")
        return behavior_data
    
    def _infer_behavior_from_emotion(self, behavior_col, annotation_data):
        """
        Infer behavior feature value from emotion if not directly provided
        
        Args:
            behavior_col: Behavior column name
            annotation_data: Annotation data for an image
            
        Returns:
            float: Inferred behavior value (0.0 or 1.0)
        """
        # Default to 0.0 (not present)
        value = 0.0
        
        # Try to get the primary emotion
        emotion = None
        if "emotions" in annotation_data and "primary_emotion" in annotation_data["emotions"]:
            emotion = annotation_data["emotions"]["primary_emotion"]
        elif "primary_emotion" in annotation_data:
            emotion = annotation_data["primary_emotion"]
        
        if emotion is None:
            return value
        
        # If we have matrix data, use it to infer behavior
        if self.matrix_data and "mappings" in self.matrix_data and behavior_col in self.matrix_data["mappings"]:
            mapping = self.matrix_data["mappings"][behavior_col]
            
            # Check if this emotion is mapped to this behavior
            if emotion in mapping and mapping[emotion] == 1:
                value = 1.0
            
            # Also check using safe emotion name
            safe_emotion = emotion.replace('/', '_')
            if safe_emotion in mapping and mapping[safe_emotion] == 1:
                value = 1.0
        else:
            # If no matrix data, use simple rules based on behavior name and emotion
            
            # Happy/Playful behaviors
            if emotion in ["Happy/Playful", "Happy_Playful"]:
                if any(term in behavior_col for term in ["tail_wagging", "mouth_open", "ears_forward", "posture_tall"]):
                    value = 1.0
            
            # Relaxed behaviors
            elif emotion in ["Relaxed"]:
                if any(term in behavior_col for term in ["posture_relaxed", "ears_neutral", "eyes_squinting"]):
                    value = 1.0
            
            # Submissive behaviors
            elif emotion in ["Submissive/Appeasement", "Submissive_Appeasement"]:
                if any(term in behavior_col for term in ["tail_low", "ears_back", "eyes_squinting", "posture_low"]):
                    value = 1.0
            
            # Curious/Alert behaviors
            elif emotion in ["Curiosity/Alertness", "Curiosity_Alertness"]:
                if any(term in behavior_col for term in ["ears_forward", "eyes_wide", "posture_tall"]):
                    value = 1.0
            
            # Stressed behaviors
            elif emotion in ["Stressed"]:
                if any(term in behavior_col for term in ["posture_stiff", "tail_low", "mouth_closed"]):
                    value = 1.0
            
            # Fearful behaviors
            elif emotion in ["Fearful/Anxious", "Fearful_Anxious"]:
                if any(term in behavior_col for term in ["tail_low", "ears_back", "eyes_wide", "posture_low"]):
                    value = 1.0
            
            # Aggressive behaviors
            elif emotion in ["Aggressive/Threatening", "Aggressive_Threatening"]:
                if any(term in behavior_col for term in ["teeth_showing", "ears_forward", "posture_stiff", "tail_high"]):
                    value = 1.0
        
        return value
    
    def create_data_generator(self, 
                              image_paths, 
                              labels, 
                              behavior_data=None,
                              img_size=(224, 224), 
                              batch_size=32, 
                              augment=False):
        """
        Create a data generator for training/validation
        
        Args:
            image_paths: List of image paths
            labels: List of class indices
            behavior_data: Dictionary mapping image paths to behavior features
            img_size: Image dimensions (height, width)
            batch_size: Batch size
            augment: Whether to apply data augmentation
            
        Returns:
            Data generator
        """
        # Convert labels to one-hot encoding
        num_classes = len(set(labels))
        labels_onehot = tf.keras.utils.to_categorical(labels, num_classes)
        
        # Get behavior size
        if behavior_data and len(behavior_data) > 0:
            first_key = next(iter(behavior_data.keys()))
            behavior_size = len(behavior_data[first_key])
        else:
            behavior_size = len(self.get_behavior_columns())
        
        # Create data generator class
        class DataGenerator(tf.keras.utils.Sequence):
            def __init__(self, image_paths, labels, behavior_data, behavior_size, img_size, batch_size, augment):
                self.image_paths = image_paths
                self.labels = labels
                self.behavior_data = behavior_data or {}
                self.behavior_size = behavior_size
                self.img_size = img_size
                self.batch_size = batch_size
                self.augment = augment
                self.indices = np.arange(len(self.image_paths))
                
                # Setup augmentation if needed
                if self.augment:
                    self.img_gen = ImageDataGenerator(
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        brightness_range=[0.8, 1.2],
                        fill_mode='nearest'
                    )
            
            def __len__(self):
                """Number of batches per epoch"""
                return int(np.ceil(len(self.image_paths) / self.batch_size))
            
            def __getitem__(self, idx):
                """Generate one batch of data"""
                batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_size = len(batch_indices)
                
                # Initialize batch arrays
                batch_images = np.zeros((batch_size, *self.img_size, 3), dtype=np.float32)
                batch_behaviors = np.zeros((batch_size, self.behavior_size), dtype=np.float32)
                batch_labels = np.zeros((batch_size, num_classes), dtype=np.float32)
                batch_confidence = np.ones((batch_size, 1), dtype=np.float32)  # Default high confidence
                
                # Fill batch data
                for i, idx in enumerate(batch_indices):
                    # Get image path
                    img_path = self.image_paths[idx]
                    
                    try:
                        # Load and preprocess image
                        img = cv2.imread(img_path)
                        if img is None:
                            raise ValueError(f"Failed to load image: {img_path}")
                        
                        # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Resize image
                        img = cv2.resize(img, self.img_size)
                        
                        # Normalize pixel values
                        img = img.astype(np.float32) / 255.0
                        
                        # Apply augmentation if enabled
                        if self.augment:
                            img = self.img_gen.random_transform(img)
                        
                        # Store image and label
                        batch_images[i] = img
                        batch_labels[i] = self.labels[idx]
                        
                        # Add behavioral data if available
                        behavior_matched = False
                        
                        # Try different matching strategies
                        matching_keys = [
                            img_path,  # Full path
                            os.path.basename(img_path),  # Basename with extension
                            os.path.splitext(os.path.basename(img_path))[0]  # Basename without extension
                        ]
                        
                        # Add numerical variations (frame_#### vs frame_00####)
                        basename = os.path.basename(img_path)
                        basename_no_ext = os.path.splitext(basename)[0]
                        
                        # Try to extract frame number or other patterns
                        frame_match = re.search(r'frame_0*(\d+)', basename_no_ext)
                        if frame_match:
                            frame_num = frame_match.group(1)
                            # Add variations with different zero padding
                            for padding in range(3, 7):  # Try 3-6 digit padding
                                matching_keys.append(f"frame_{int(frame_num):0{padding}d}")
                        
                        # Try each key
                        for key in matching_keys:
                            if key in self.behavior_data:
                                batch_behaviors[i] = self.behavior_data[key]
                                behavior_matched = True
                                break
                        
                        # If behavior data wasn't found, count how many behavior features are missing
                        if not behavior_matched:
                            # Lower confidence slightly if behaviors missing
                            batch_confidence[i] = 0.8
                    
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
                        # Use zeros for this sample, but mark very low confidence
                        batch_confidence[i] = 0.2
                
                # Return the batch
                inputs = {
                    'image_input': batch_images,
                    'behavior_input': batch_behaviors
                }
                
                outputs = {
                    'emotion_output': batch_labels,
                    'confidence_output': batch_confidence
                }
                
                return inputs, outputs
            
            def on_epoch_end(self):
                """Shuffle indices after each epoch"""
                if self.augment:  # Only shuffle training data
                    np.random.shuffle(self.indices)
        
        # Create and return data generator
        return DataGenerator(image_paths, labels_onehot, behavior_data, behavior_size, 
                            img_size, batch_size, augment)
    
    def train(self, epochs=50, batch_size=32, img_size=(224, 224), fine_tune=True):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image dimensions (height, width)
            fine_tune: Whether to perform fine-tuning
            
        Returns:
            tuple: (history, model_dir)
        """
        # Load data from annotations
        print("Loading data from annotations...")
        train_paths, train_labels, class_names, train_ids = self.load_data_from_annotations('train')
        val_paths, val_labels, _, val_ids = self.load_data_from_annotations('validation')
        
        # Load annotations to extract behavior features
        train_annotations = self.load_annotations('train')
        
        # Extract behavior features
        behavior_data = self.extract_behavior_features(train_annotations, train_ids + val_ids)
        
        # Create data generators
        train_gen = self.create_data_generator(
            train_paths, train_labels, behavior_data, img_size, batch_size, augment=True)
        
        val_gen = self.create_data_generator(
            val_paths, val_labels, behavior_data, img_size, batch_size, augment=False)
        
        # Create model if not already created
        if self.model is None:
            self.create_model(
                num_classes=len(class_names),
                behavior_size=len(self.get_behavior_columns()),
                img_size=(*img_size, 3)
            )
        
        # Create model directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = os.path.join(self.model_dir, f"dog_emotion_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create callbacks
        callbacks = [
            
            # Early stopping
            EarlyStopping(
                monitor='val_emotion_output_accuracy',
                patience=10,
                restore_best_weights=True,
                mode='max'  # Add this line to specify we want to maximize accuracy
            ),
            # Model checkpoints
            ModelCheckpoint(
                os.path.join(model_dir, 'best_model.h5'),
                monitor='val_emotion_output_accuracy',
                save_best_only=True,
                mode='max'
            ),
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_emotion_output_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode='max'
            )
        ]
        
        # Train model
        print(f"Training model with {len(train_paths)} images and {len(self.get_behavior_columns())} behavior features")
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks
        )
        
        # Fine-tuning if requested
        if fine_tune:
            print("Fine-tuning the model...")
            
            # Fine-tune the model
            self.fine_tune_model(unfreeze_layers=15)
            
            # Train with fine-tuning
            ft_history = self.model.fit(
                train_gen,
                epochs=20,  # Fewer epochs for fine-tuning
                validation_data=val_gen,
                callbacks=callbacks
            )
            
            # Extend history
            for k in history.history:
                if k in ft_history.history:
                    history.history[k].extend(ft_history.history[k])
        
        # Save final model
        self.model.save(os.path.join(model_dir, 'final_model.h5'))
        
        # Save class names and behavior columns
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump({
                'class_names': class_names,
                'behavior_columns': self.get_behavior_columns(),
                'img_size': list(img_size),
                'training_timestamp': timestamp
            }, f, indent=2)
        
        # Save training history
        history_dict = {}
        for k, v in history.history.items():
            history_dict[k] = [float(x) for x in v]
            
        with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        # Plot training history
        self.plot_training_history(history, model_dir)
        
        # Evaluate on test set
        self.evaluate(model_dir)
        
        return history, model_dir
    
    def evaluate(self, model_dir):
        """
        Evaluate the model on the test set
        
        Args:
            model_dir: Directory to save evaluation results
        """
        # Load test data
        test_paths, test_labels, _, test_ids = self.load_data_from_annotations('test')
        
        # Load annotations to extract behavior features
        test_annotations = self.load_annotations('test')
        behavior_data = self.extract_behavior_features(test_annotations, test_ids)
        
        # Create test generator
        test_gen = self.create_data_generator(
            test_paths, test_labels, behavior_data, (224, 224), batch_size=32, augment=False)
        
        # Evaluate model
        print("Evaluating model on test set...")
        evaluation = self.model.evaluate(test_gen)
        
        # Get metric names
        metric_names = self.model.metrics_names
        
        # Save evaluation metrics
        metrics = {}
        for i, metric_name in enumerate(metric_names):
            metrics[metric_name] = float(evaluation[i])
        
        with open(os.path.join(model_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate predictions
        print("Generating predictions for confusion matrix...")
        y_true = []
        y_pred = []
        
        for i in range(len(test_gen)):
            # Get batch data
            inputs, outputs = test_gen[i]
            
            # True labels
            batch_y = outputs['emotion_output']
            batch_y_true = np.argmax(batch_y, axis=1)
            
            # Predict batch
            batch_pred = self.model.predict(inputs)
            batch_y_pred = np.argmax(batch_pred[0], axis=1)  # First output is emotion
            
            # Add to lists
            y_true.extend(batch_y_true)
            y_pred.extend(batch_y_pred)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Normalized Confusion Matrix')
        plt.colorbar()
        
        class_names = self.class_names
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        fmt = '.2f'
        thresh = cm_norm.max() / 2.
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(j, i, format(cm_norm[i, j], fmt),
                         ha="center", va="center",
                         color="white" if cm_norm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save confusion matrix
        plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True)
        
        with open(os.path.join(model_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"Test metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        print("Classification report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
    
    def plot_training_history(self, history, output_dir):
        """
        Plot training history
        
        Args:
            history: Training history
            output_dir: Directory to save plots
        """
        # Plot emotion accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(history.history.get('emotion_output_accuracy', []), label='Training')
        plt.plot(history.history.get('val_emotion_output_accuracy', []), label='Validation')
        plt.title('Model Emotion Classification Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'emotion_accuracy.png'), dpi=300)
        plt.close()
        
        # Plot emotion loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history.get('emotion_output_loss', []), label='Training')
        plt.plot(history.history.get('val_emotion_output_loss', []), label='Validation')
        plt.title('Model Emotion Classification Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'emotion_loss.png'), dpi=300)
        plt.close()
        
        # Plot confidence accuracy
        if 'confidence_output_accuracy' in history.history:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['confidence_output_accuracy'], label='Training')
            plt.plot(history.history['val_confidence_output_accuracy'], label='Validation')
            plt.title('Model Confidence Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'confidence_accuracy.png'), dpi=300)
            plt.close()
        
        # Plot confidence loss
        if 'confidence_output_loss' in history.history:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['confidence_output_loss'], label='Training')
            plt.plot(history.history['val_confidence_output_loss'], label='Validation')
            plt.title('Model Confidence Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'confidence_loss.png'), dpi=300)
            plt.close()
        
        # Plot combined loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history.get('loss', []), label='Training')
        plt.plot(history.history.get('val_loss', []), label='Validation')
        plt.title('Model Combined Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'combined_loss.png'), dpi=300)
        plt.close()
    
    def predict_image(self, image_path, behavior_features=None):
        """
        Predict emotion for a single image
        
        Args:
            image_path: Path to image file
            behavior_features: Optional list of behavior features
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Load and preprocess image
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, (224, 224))
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            # Create behavior features if not provided
            if behavior_features is None:
                behavior_size = len(self.get_behavior_columns())
                behavior_features = np.zeros((1, behavior_size), dtype=np.float32)
            else:
                behavior_features = np.array([behavior_features], dtype=np.float32)
            
            # Make prediction
            inputs = {
                'image_input': img,
                'behavior_input': behavior_features
            }
            predictions = self.model.predict(inputs)
            
            # Get emotion predictions and confidence
            emotion_preds = predictions[0]  # First output is emotion
            confidence_pred = predictions[1]  # Second output is confidence
            
            # Get the predicted emotion
            predicted_idx = np.argmax(emotion_preds[0])
            emotion_score = float(emotion_preds[0][predicted_idx])
            confidence_score = float(confidence_pred[0][0])
            
            if len(self.class_names) > predicted_idx:
                predicted_class = self.class_names[predicted_idx]
            else:
                predicted_class = f"Class {predicted_idx}"
            
            # Create all predictions dict
            all_emotions = {
                self.class_names[i]: float(emotion_preds[0][i]) 
                for i in range(len(self.class_names))
            }
            
            # Map safe class name to standard name if needed
            if predicted_class in self.safe_to_standard:
                standard_emotion = self.safe_to_standard[predicted_class]
            else:
                standard_emotion = predicted_class
            
            # Create result dictionary
            result = {
                'emotion': standard_emotion,
                'emotion_score': emotion_score,
                'confidence': confidence_score,
                'all_emotions': all_emotions,
                'behavior_features': behavior_features[0].tolist()
            }
            
            return result
        
        except Exception as e:
            print(f"Error predicting image: {str(e)}")
            return None
    
    def load_model(self, model_path, metadata_path=None):
        """
        Load a saved model
        
        Args:
            model_path: Path to saved model file (.h5)
            metadata_path: Path to model metadata file (.json)
            
        Returns:
            Loaded model
        """
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            
            # If metadata_path not provided, try to find it in the same directory
            if metadata_path is None:
                model_dir = os.path.dirname(model_path)
                potential_metadata = os.path.join(model_dir, 'model_metadata.json')
                if os.path.exists(potential_metadata):
                    metadata_path = potential_metadata
            
            # Load metadata if provided
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Extract metadata
                self.class_names = metadata.get('class_names', [])
                self.behavior_columns = metadata.get('behavior_columns', [])
                
                print(f"Loaded metadata: {len(self.class_names)} classes, {len(self.behavior_columns)} behavior features")
            else:
                print("No metadata file found. Using default class names and behavior columns.")
                # Set default class names if none loaded
                if not self.class_names:
                    self.class_names = self.safe_class_names
            
            return self.model
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def visualize_prediction(self, image_path, result, output_path=None):
        """
        Visualize prediction results
        
        Args:
            image_path: Path to image file
            result: Prediction result dictionary
            output_path: Optional path to save visualization
            
        Returns:
            None
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Image subplot
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Predicted: {result['emotion']} ({result['emotion_score']:.2f})")
        plt.axis('off')
        
        # Prediction subplot
        plt.subplot(1, 2, 2)
        
        # Sort emotions by score
        emotions = []
        scores = []
        for emotion, score in sorted(result['all_emotions'].items(), key=lambda x: x[1], reverse=True):
            emotions.append(emotion)
            scores.append(score)
        
        # Bar chart
        bars = plt.barh(emotions, scores, color='skyblue')
        
        # Highlight predicted emotion
        for i, emotion in enumerate(emotions):
            if emotion == self.class_name_mapping.get(result['emotion'], result['emotion']):
                bars[i].set_color('green')
        
        # Add confidence line
        plt.axvline(x=result['confidence'], color='red', linestyle='--', 
                   label=f"Confidence: {result['confidence']:.2f}")
        
        plt.xlabel('Score')
        plt.ylabel('Emotion')
        plt.title('Emotion Predictions')
        plt.xlim(0, 1)
        plt.legend()
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"Visualization saved to {output_path}")
        
        plt.show()

# Usage example
if __name__ == "__main__":
    # Create classifier
    classifier = DogEmotionWithBehaviors()
    
    # Train model
    history, model_dir = classifier.train(
        epochs=50,
        batch_size=32,
        fine_tune=True
    )
    
    print(f"Training completed. Model saved to {model_dir}")