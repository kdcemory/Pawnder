# Updated Dog Emotion Model Training Pipeline
# This script handles data loading, augmentation, and model training
# Incorporates changes from other project components

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
)
from tensorflow.keras.optimizers import Adam
import cv2
import json
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import logging
IN_NOTEBOOK = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DogEmotionTrainer")

# Import the model builder - placeholder for the actual import
# from dog_emotion_model import DogEmotionModel

class DogEmotionModel:
    """Model architecture for dog emotion recognition"""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the model builder
        
        Args:
            config_path (str): Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
    def create_model(self, model_type='image', num_emotions=14, num_behaviors=20):
        """
        Create a dog emotion recognition model
        
        Args:
            model_type (str): 'image' for image-based or 'video' for video-based model
            num_emotions (int): Number of output emotion classes
            num_behaviors (int): Number of behavioral indicator features
            
        Returns:
            tf.keras.Model: Compiled model
        """
        # Get configuration
        img_size = tuple(self.config['model']['image_size'])
        dropout_rate = self.config['model']['dropout_rate']
        learning_rate = self.config['model']['learning_rate']
        backbone = self.config['model']['backbone']
        
        # Create image input
        image_input = tf.keras.layers.Input(shape=img_size, name='image_input')
        
        # Create behavior input
        behavior_input = tf.keras.layers.Input(shape=(num_behaviors,), name='behavior_input')
        
        # Feature extraction backbone
        if backbone == 'mobilenetv2':
            # Use MobileNetV2 as backbone
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=img_size,
                include_top=False,
                weights='imagenet'
            )
        elif backbone == 'resnet50':
            # Use ResNet50 as backbone
            base_model = tf.keras.applications.ResNet50(
                input_shape=img_size,
                include_top=False,
                weights='imagenet'
            )
        elif backbone == 'efficientnetb0':
            # Use EfficientNetB0 as backbone
            base_model = tf.keras.applications.EfficientNetB0(
                input_shape=img_size,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze base model layers for initial training
        for layer in base_model.layers:
            layer.trainable = False
        
        # Extract features
        x = base_model(image_input, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        
        # Additional video features (only for video model)
        if model_type == 'video':
            # Placeholder for temporal features integration
            # This could be implemented with LSTM/GRU layers if needed
            pass
        
        # Combine with behavioral features
        behavior_features = tf.keras.layers.Dense(64, activation='relu')(behavior_input)
        behavior_features = tf.keras.layers.Dropout(dropout_rate/2)(behavior_features)
        
        # Concatenate image and behavior features
        combined_features = tf.keras.layers.Concatenate()([x, behavior_features])
        
        # Dense layers for classification
        x = tf.keras.layers.Dense(128, activation='relu')(combined_features)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        # Output layers
        emotion_output = tf.keras.layers.Dense(num_emotions, activation='softmax', name='emotion_output')(x)
        confidence_output = tf.keras.layers.Dense(1, activation='sigmoid', name='confidence_output')(x)
        
        # Create model
        model = tf.keras.Model(
            inputs=[image_input, behavior_input],
            outputs=[emotion_output, confidence_output]
        )
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                'emotion_output': 'categorical_crossentropy',
                'confidence_output': 'binary_crossentropy'
            },
            metrics={
                'emotion_output': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')],
                'confidence_output': ['accuracy']
            },
            loss_weights={
                'emotion_output': 1.0,
                'confidence_output': 0.2
            }
        )
        
        return model
    
    def fine_tune_model(self, model, num_layers=10):
        """
        Fine-tune the model by unfreezing some backbone layers
        
        Args:
            model (tf.keras.Model): Trained model
            num_layers (int): Number of layers to unfreeze from the end
            
        Returns:
            tf.keras.Model: Model ready for fine-tuning
        """
        # Find the base model layers
        for layer in model.layers:
            if hasattr(layer, 'layers'):  # Check if layer is a model
                base_model = layer
                break
        
        # Get the total number of layers
        total_layers = len(base_model.layers)
        
        # Unfreeze the last n layers
        for layer in base_model.layers[-(num_layers):]:
            layer.trainable = True
        
        # Use a lower learning rate for fine-tuning
        lr = self.config['model']['learning_rate'] / 10
        
        # Recompile the model
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss={
                'emotion_output': 'categorical_crossentropy',
                'confidence_output': 'binary_crossentropy'
            },
            metrics={
                'emotion_output': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')],
                'confidence_output': ['accuracy']
            },
            loss_weights={
                'emotion_output': 1.0,
                'confidence_output': 0.2
            }
        )
        
        return model


class DogEmotionDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for dog emotion recognition model"""
    
    def __init__(self, annotations_df, img_dir, behavior_cols, emotion_col, 
                 img_size=(224, 224), batch_size=32, shuffle=True, augment=False,
                 use_bbox=True, unknown_as_none=False):
        """
        Initialize the data generator
        
        Args:
            annotations_df (pd.DataFrame): DataFrame with annotations
            img_dir (str): Directory containing the images
            behavior_cols (list): List of behavioral indicator columns
            emotion_col (str): Column name for emotion labels
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle data between epochs
            augment (bool): Whether to apply data augmentation
            use_bbox (bool): Whether to use bounding boxes for cropping
            unknown_as_none (bool): Treat "Unknown" emotion as None and reduce confidence
        """
        self.annotations = annotations_df
        self.img_dir = img_dir
        self.behavior_cols = behavior_cols
        self.emotion_col = emotion_col
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.use_bbox = use_bbox
        self.unknown_as_none = unknown_as_none
        
        # Get unique emotions for one-hot encoding
        all_emotions = self.annotations[self.emotion_col].dropna().unique()
        # Filter out "Unknown" if needed
        if self.unknown_as_none:
            self.emotions = [e for e in all_emotions if e.lower() != "unknown"]
        else:
            self.emotions = all_emotions
            
        self.emotion_to_idx = {emotion: i for i, emotion in enumerate(self.emotions)}
        
        # Create indices and shuffle if needed
        self.indices = np.arange(len(self.annotations))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Set up augmentation if needed
        if self.augment:
            self.img_gen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
    
    def __len__(self):
        """Return the number of batches per epoch"""
        return int(np.ceil(len(self.annotations) / self.batch_size))
    
    def __getitem__(self, idx):
        """Generate one batch of data"""
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.annotations.iloc[batch_indices]
        
        # Initialize batch arrays
        batch_images = np.zeros((len(batch_data), *self.img_size, 3), dtype=np.float32)
        batch_behaviors = np.zeros((len(batch_data), len(self.behavior_cols)), dtype=np.float32)
        batch_emotions = np.zeros((len(batch_data), len(self.emotions)), dtype=np.float32)
        batch_confidence = np.ones((len(batch_data), 1), dtype=np.float32)  # Default confidence of 1.0
        
        # Fill batch data
        for i, (_, row) in enumerate(batch_data.iterrows()):
            # Load and preprocess image
            img_path = os.path.join(self.img_dir, row['image_path'])
            try:
                img = self._load_and_preprocess_image(img_path, row)
                batch_images[i] = img
                
                # Extract behavioral indicators
                for j, col in enumerate(self.behavior_cols):
                    batch_behaviors[i, j] = row[col] if col in row and not pd.isna(row[col]) else 0
                
                # One-hot encode emotion
                emotion = row[self.emotion_col]
                if pd.isna(emotion) or (self.unknown_as_none and emotion.lower() == "unknown"):
                    # No clear emotion, use lower confidence
                    batch_confidence[i] = 0.5
                    # Use a balanced distribution of emotions
                    batch_emotions[i] = np.ones(len(self.emotions)) / len(self.emotions)
                else:
                    emotion_idx = self.emotion_to_idx.get(emotion, -1)
                    if emotion_idx >= 0:
                        batch_emotions[i, emotion_idx] = 1.0
                    else:
                        # Unknown emotion, use lower confidence
                        batch_confidence[i] = 0.7
                        # Use a balanced distribution
                        batch_emotions[i] = np.ones(len(self.emotions)) / len(self.emotions)
                
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                # If image loading fails, use zeros and lower confidence
                batch_confidence[i] = 0.3
                # Use a balanced distribution for emotions
                batch_emotions[i] = np.ones(len(self.emotions)) / len(self.emotions)
        
        # Return inputs and outputs
        inputs = {'image_input': batch_images, 'behavior_input': batch_behaviors}
        outputs = {'emotion_output': batch_emotions, 'confidence_output': batch_confidence}
        
        return inputs, outputs
    
    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch if needed"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_and_preprocess_image(self, img_path, row):
        """Load and preprocess an image"""
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop to bounding box if available and enabled
        if self.use_bbox and all(col in row.index and not pd.isna(row[col]) 
                               for col in ['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']):
            x, y = int(row['bbox_x']), int(row['bbox_y'])
            w, h = int(row['bbox_width']), int(row['bbox_height'])
            
            # Ensure bbox is within image bounds
            img_h, img_w = img.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            # Crop if bbox is valid
            if w > 10 and h > 10:  # Ensure minimal size to avoid tiny crops
                img = img[y:y+h, x:x+w]
            else:
                logger.warning(f"Invalid bounding box dimensions: {w}x{h} for {img_path}")
        
        # Resize to target size
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Apply augmentation if needed
        if self.augment:
            img = self.img_gen.random_transform(img)
        
        return img


class DogEmotionTrainer:
    """Trainer for Dog Emotion Recognition model"""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the trainer
        
        Args:
            config_path (str): Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.model_builder = DogEmotionModel(config_path)
        self.training_history = None
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    def load_annotations(self, split='train'):
        """
        Load annotation data for a specific split
        
        Args:
            split (str): 'train', 'val', or 'test'
            
        Returns:
            pd.DataFrame: Loaded annotations
        """
        annotations_path = os.path.join(
            self.config['data']['base_dir'],
            self.config['data']['processed_data_dir'],
            split,
            'annotations.csv'
        )
        
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")
        
        df = pd.read_csv(annotations_path)
        logger.info(f"Loaded {len(df)} annotations for {split} split")
        return df
    
    def load_annotations_with_reduced_emotions(self, split='train'):
        """
        Load annotation data with reduced emotion set for a specific split
        
        Args:
            split (str): 'train', 'val', or 'test'
          
        Returns:
            pd.DataFrame: Loaded annotations with reduced emotions
        """
        # Load the original annotations
        df = self.load_annotations(split)
        
        # Map to reduced emotion set
        df = map_emotions_to_reduced_set(df)
        
        # Use the reduced emotion column instead of the original
        logger.info(f"Reduced emotions in {split} split: {df['reduced_emotional_state'].unique()}")
        logger.info(f"Emotion counts in {split}: {df['reduced_emotional_state'].value_counts()}")
        
        return df
        
    def get_behavior_columns(self, df):
        """
        Get columns containing behavioral indicators
        
        Args:
            df (pd.DataFrame): Annotations DataFrame
            
        Returns:
            list: Behavioral indicator column names
        """
        # Find columns that are behavior indicators (assuming they start with 'behavior_')
        behavior_cols = [col for col in df.columns if col.startswith('behavior_')]
        if not behavior_cols:
            # Fallback: look for any columns that might contain behavioral data
            behavior_cols = [col for col in df.columns if any(term in col.lower() for term 
                                                            in ['behavior', 'posture', 'ear', 'tail', 'mouth', 'eye'])]
        
        logger.info(f"Identified {len(behavior_cols)} behavioral feature columns")
        return behavior_cols
    
    def setup_data_generators(self, img_size=(224, 224), batch_size=32):
        """
        Set up data generators for training and validation
        
        Args:
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size
            
        Returns:
            tuple: (train_generator, val_generator)
        """
        # Load annotations
        train_df = self.load_annotations('train')
        val_df = self.load_annotations('val')
        
        # Get behavior columns
        behavior_cols = self.get_behavior_columns(train_df)
        
        # Define directories
        train_img_dir = os.path.join(
            self.config['data']['base_dir'],
            self.config['data']['processed_data_dir'],
            'train',
            'images'
        )
        
        val_img_dir = os.path.join(
            self.config['data']['base_dir'],
            self.config['data']['processed_data_dir'],
            'val',
            'images'
        )
        
        # Augmentation settings
        augment = self.config['training']['augmentation'].get('enable', True)
        
        # Create generators
        train_generator = DogEmotionDataGenerator(
            train_df,
            train_img_dir,
            behavior_cols,
            'emotional_state',
            img_size=img_size,
            batch_size=batch_size,
            shuffle=True,
            augment=augment,
            use_bbox=True,
            unknown_as_none=True
        )
        
        val_generator = DogEmotionDataGenerator(
            val_df,
            val_img_dir,
            behavior_cols,
            'emotional_state',
            img_size=img_size,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
            use_bbox=True,
            unknown_as_none=True
        )
        
        return train_generator, val_generator, behavior_cols
    
    def setup_callbacks(self, model_name):
        """
        Set up training callbacks
        
        Args:
            model_name (str): Name for the model
            
        Returns:
            list: Callbacks for training
        """
        # Create timestamp for unique folder names
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Define directories
        checkpoint_dir = os.path.join(
            self.config['data']['base_dir'],
            self.config['training']['checkpoint_dir'],
            f"{model_name}_{timestamp}"
        )
        
        logs_dir = os.path.join(
            self.config['data']['base_dir'],
            self.config['training']['logs_dir'],
            f"{model_name}_{timestamp}"
        )
        
        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # CSV logger path
        csv_log_path = os.path.join(checkpoint_dir, "training_log.csv")
        
        # Define callbacks
        callbacks = [
            # Stop training when validation loss stops improving
            EarlyStopping(
                monitor='val_emotion_output_accuracy',
                patience=self.config['model']['early_stopping_patience'],
                mode='max',
                verbose=1,
                restore_best_weights=True
            ),
            
            # Reduce learning rate when validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_emotion_output_accuracy',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                mode='max',
                verbose=1
            ),
            
            # Save the best model
            ModelCheckpoint(
                os.path.join(checkpoint_dir, 'best_model.h5'),
                monitor='val_emotion_output_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Save regular checkpoints
            ModelCheckpoint(
                os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.h5'),
                save_freq='epoch',
                verbose=0
            ),
            
            # TensorBoard logs
            TensorBoard(
                log_dir=logs_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            
            # CSV Logger
            CSVLogger(csv_log_path, append=True)
        ]
        
        return callbacks, checkpoint_dir
    
    def train_model(self, model_type='image', epochs=50, learning_rate=None, 
                   fine_tune=True, fine_tune_epochs=20, fine_tune_layers=15):
        """
        Train the model
        
        Args:
            model_type (str): 'image' or 'video'
            epochs (int): Number of training epochs
            learning_rate (float): Optional custom learning rate
            fine_tune (bool): Whether to fine-tune the model after initial training
            fine_tune_epochs (int): Number of fine-tuning epochs
            fine_tune_layers (int): Number of layers to unfreeze for fine-tuning
            
        Returns:
            tuple: (trained_model, history, checkpoint_dir)
        """
        # Get model configuration
        img_size = tuple(self.config['model']['image_size'])
        batch_size = self.config['model']['batch_size']
        
        # Override learning rate if provided
        if learning_rate is not None:
            self.config['model']['learning_rate'] = learning_rate
        
        # Set up data generators
        train_generator, val_generator, behavior_cols = self.setup_data_generators(
            img_size=img_size[:2],  # Height, width only
            batch_size=batch_size
        )
        
        # Build the model
        num_emotions = len(train_generator.emotions)
        num_behaviors = len(behavior_cols)
        
        logger.info(f"Creating model with {num_emotions} emotion classes and {num_behaviors} behavioral features")
        model = self.model_builder.create_model(
            model_type=model_type,
            num_emotions=num_emotions,
            num_behaviors=num_behaviors
        )
        
        # Set up callbacks
        callbacks, checkpoint_dir = self.setup_callbacks(f"dog_emotion_{model_type}")
        
        # Save emotion mapping
        emotion_mapping = {i: emotion for emotion, i in train_generator.emotion_to_idx.items()}
        with open(os.path.join(checkpoint_dir, 'emotion_mapping.json'), 'w') as f:
            json.dump(emotion_mapping, f, indent=2)
        
        # Save behavior columns
        with open(os.path.join(checkpoint_dir, 'behavior_columns.json'), 'w') as f:
            json.dump(behavior_cols, f, indent=2)
        
        # Print model summary
        model.summary()
        
        # Train the model
        logger.info(f"Starting training for {model_type}-based model...")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tune if requested
        if fine_tune:
            logger.info(f"Fine-tuning model by unfreezing {fine_tune_layers} layers...")
            
            # Fine-tune the model by unfreezing some backbone layers
            model = self.model_builder.fine_tune_model(model, num_layers=fine_tune_layers)
            
            # Update callbacks for fine-tuning
            ft_callbacks, _ = self.setup_callbacks(f"dog_emotion_{model_type}_finetune")
            
            # Train with fine-tuning
            ft_history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=fine_tune_epochs,
                callbacks=ft_callbacks,
                verbose=1
            )
            
            # Combine histories
            for k in history.history:
                history.history[k].extend(ft_history.history[k])
        
        # Save the final model
        final_model_path = os.path.join(
            checkpoint_dir,
            f"dog_emotion_{model_type}_final.h5"
        )
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Save training history
        history_path = os.path.join(
            checkpoint_dir,
            f"dog_emotion_{model_type}_history.json"
        )
        
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {}
            for key, values in history.history.items():
                history_dict[key] = [float(val) for val in values]
            
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
        
        # Save training metadata
        metadata = {
            'model_type': model_type,
            'num_emotions': num_emotions,
            'num_behaviors': num_behaviors,
            'emotions': list(train_generator.emotion_to_idx.keys()),
            'behavior_columns': behavior_cols,
            'img_size': img_size,
            'backbone': self.config['model']['backbone'],
            'total_epochs': epochs + (fine_tune_epochs if fine_tune else 0),
            'fine_tuned': fine_tune,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(checkpoint_dir, 'training_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Store training history for later use
        self.training_history = history
        
        return model, history, checkpoint_dir
    
    def evaluate_model(self, model, model_type='image', checkpoint_dir=None):
        """
        Evaluate the trained model on the test set
        
        Args:
            model (tf.keras.Model): Trained model
            model_type (str): 'image' or 'video'
            checkpoint_dir (str): Directory to save evaluation results
            
        Returns:
            dict: Evaluation metrics
        """
        if checkpoint_dir is None:
            # Use latest checkpoint directory if not provided
            base_dir = os.path.join(
                self.config['data']['base_dir'],
                self.config['training']['checkpoint_dir']
            )
            
            # Find directories with the model_type in the name
            dirs = [d for d in os.listdir(base_dir) if model_type in d and os.path.isdir(os.path.join(base_dir, d))]
            
            if not dirs:
                raise ValueError(f"No checkpoint directory found for {model_type} model")
            
            # Use the most recent directory
            checkpoint_dir = os.path.join(base_dir, sorted(dirs)[-1])
        
        # Load test data
        test_df = self.load_annotations('test')
        behavior_cols = self.get_behavior_columns(test_df)
        
        # Load emotion mapping
        emotion_mapping_path = os.path.join(checkpoint_dir, 'emotion_mapping.json')
        if os.path.exists(emotion_mapping_path):
            with open(emotion_mapping_path, 'r') as f:
                emotion_mapping = json.load(f)
                emotions = list(emotion_mapping.values())
        else:
            # Extract emotions from test data
            emotions = test_df['emotional_state'].dropna().unique().tolist()
        
        # Set up test generator
        test_img_dir = os.path.join(
            self.config['data']['base_dir'],
            self.config['data']['processed_data_dir'],
            'test',
            'images'
        )
        
        img_size = tuple(self.config['model']['image_size'])
        batch_size = self.config['model']['batch_size']
        
        test_generator = DogEmotionDataGenerator(
            test_df,
            test_img_dir,
            behavior_cols,
            'emotional_state',
            img_size=img_size[:2],  # Height, width only
            batch_size=batch_size,
            shuffle=False,
            augment=False,
            use_bbox=True
        )
        
        # Evaluate the model
        logger.info(f"Evaluating {model_type}-based model on test set...")
        evaluation = model.evaluate(test_generator, verbose=1)
        
        # Get evaluation metrics
        metric_names = model.metrics_names
        evaluation_dict = {name: float(value) for name, value in zip(metric_names, evaluation)}
        
        # Print evaluation results
        logger.info("\nEvaluation Results:")
        for name, value in evaluation_dict.items():
            logger.info(f"{name}: {value:.4f}")
        
        # Save evaluation results
        eval_path = os.path.join(
            checkpoint_dir,
            f"dog_emotion_{model_type}_evaluation.json"
        )
        
        with open(eval_path, 'w') as f:
            json.dump(evaluation_dict, f, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_path}")
        
        # Generate confusion matrix and classification report
        self.generate_classification_metrics(model, test_generator, model_type, checkpoint_dir)
        
        return evaluation_dict
    
    def generate_classification_metrics(self, model, test_generator, model_type, checkpoint_dir):
        """
        Generate and save confusion matrix and classification report
        
        Args:
            model (tf.keras.Model): Trained model
            test_generator (DogEmotionDataGenerator): Test data generator
            model_type (str): 'image' or 'video'
            checkpoint_dir (str): Directory to save results
        """
        # Get predictions
        y_true = []
        y_pred = []
        y_scores = []
        y_confidence = []
        
        logger.info("Generating predictions for classification metrics...")
        for i in tqdm(range(len(test_generator))):
            inputs, outputs = test_generator[i]
            batch_pred = model.predict(inputs, verbose=0)
            
            # Extract true and predicted emotions
            batch_true = np.argmax(outputs['emotion_output'], axis=1)
            batch_pred = np.argmax(batch_pred[0], axis=1)  # First output is emotion probabilities
            
            # Store prediction scores and confidence
            y_scores.extend(batch_pred[0])
            y_confidence.extend(batch_pred[1])
            
            y_true.extend(batch_true)
            y_pred.extend(batch_pred)
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Get class names
        class_names = list(test_generator.emotion_to_idx.keys())
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Normalized Confusion Matrix - {model_type.capitalize()} Model')
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        
        # Save confusion matrix plot
        cm_path = os.path.join(
            checkpoint_dir,
            f"dog_emotion_{model_type}_confusion_matrix.png"
        )
        plt.savefig(cm_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Generate classification report
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average='weighted')
        logger.info(f"Weighted F1 Score: {f1:.4f}")
        
        # Add F1 score to report
        report['weighted_f1'] = float(f1)
        
        # Save classification report
        report_path = os.path.join(
            checkpoint_dir,
            f"dog_emotion_{model_type}_classification_report.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Calculate per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            # True positives, false positives, etc.
            true_pos = np.sum((y_true == i) & (y_pred == i))
            false_pos = np.sum((y_true != i) & (y_pred == i))
            false_neg = np.sum((y_true == i) & (y_pred != i))
            true_neg = np.sum((y_true != i) & (y_pred != i))
            
            # Calculate metrics
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(np.sum(y_true == i))
            }
        
        # Save per-class metrics
        metrics_path = os.path.join(
            checkpoint_dir,
            f"dog_emotion_{model_type}_class_metrics.json"
        )
        
        with open(metrics_path, 'w') as f:
            json.dump(class_metrics, f, indent=2)
        
        logger.info(f"Confusion matrix saved to {cm_path}")
        logger.info(f"Classification report saved to {report_path}")
        logger.info(f"Per-class metrics saved to {metrics_path}")
    
    def plot_training_history(self, history, model_type, checkpoint_dir=None):
        """
        Plot and save training history
        
        Args:
            history (tf.keras.callbacks.History): Training history
            model_type (str): 'image' or 'video'
            checkpoint_dir (str): Directory to save plot
        """
        if history is None:
            if self.training_history is None:
                logger.warning("No training history available to plot")
                return
            history = self.training_history
        
        if checkpoint_dir is None:
            # Use config directory
            checkpoint_dir = os.path.join(
                self.config['data']['base_dir'],
                self.config['training']['checkpoint_dir']
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create a more comprehensive visualization
        plt.figure(figsize=(15, 10))
        
        # Plot accuracy
        plt.subplot(2, 2, 1)
        plt.plot(history.history['emotion_output_accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_emotion_output_accuracy'], label='Validation Accuracy')
        plt.title('Emotion Recognition Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(2, 2, 2)
        plt.plot(history.history['emotion_output_loss'], label='Training Loss')
        plt.plot(history.history['val_emotion_output_loss'], label='Validation Loss')
        plt.title('Emotion Recognition Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot top-3 accuracy if available
        if 'emotion_output_top3_acc' in history.history:
            plt.subplot(2, 2, 3)
            plt.plot(history.history['emotion_output_top3_acc'], label='Training Top-3 Accuracy')
            plt.plot(history.history['val_emotion_output_top3_acc'], label='Validation Top-3 Accuracy')
            plt.title('Top-3 Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Top-3 Accuracy')
            plt.legend()
        
        # Plot confidence accuracy
        plt.subplot(2, 2, 4)
        plt.plot(history.history['confidence_output_accuracy'], label='Training Confidence')
        plt.plot(history.history['val_confidence_output_accuracy'], label='Validation Confidence')
        plt.title('Confidence Prediction Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        history_plot_path = os.path.join(
            checkpoint_dir,
            f"dog_emotion_{model_type}_training_history.png"
        )
        plt.savefig(history_plot_path, dpi=300)
        plt.close()
        
        # Create learning rate plot if available
        if 'lr' in history.history:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['lr'])
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
            
            lr_plot_path = os.path.join(
                checkpoint_dir,
                f"dog_emotion_{model_type}_learning_rate.png"
            )
            plt.savefig(lr_plot_path, dpi=300)
            plt.close()
            logger.info(f"Learning rate plot saved to {lr_plot_path}")
        
        logger.info(f"Training history plot saved to {history_plot_path}")
    
    # Add methods for model analysis and explanation
    def analyze_model_performance(self, checkpoint_dir):
        """
        Analyze model performance across different emotions and behaviors
        
        Args:
            checkpoint_dir (str): Directory containing evaluation results
        """
        # Load class metrics
        metrics_paths = [f for f in os.listdir(checkpoint_dir) if f.endswith('_class_metrics.json')]
        
        if not metrics_paths:
            logger.warning(f"No class metrics found in {checkpoint_dir}")
            return
        
        # Load the first metrics file
        metrics_path = os.path.join(checkpoint_dir, metrics_paths[0])
        with open(metrics_path, 'r') as f:
            class_metrics = json.load(f)
        
        # Create a bar chart of F1 scores by emotion class
        emotions = list(class_metrics.keys())
        f1_scores = [metrics['f1'] for metrics in class_metrics.values()]
        support = [metrics['support'] for metrics in class_metrics.values()]
        
        # Sort by F1 score
        sorted_indices = np.argsort(f1_scores)
        emotions = [emotions[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]
        support = [support[i] for i in sorted_indices]
        
        # Plot F1 scores
        plt.figure(figsize=(12, 8))
        bars = plt.barh(emotions, f1_scores, color='skyblue')
        
        # Add support numbers
        for i, (bar, sup) in enumerate(zip(bars, support)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"n={sup}", va='center')
        
        plt.xlabel('F1 Score')
        plt.ylabel('Emotion')
        plt.title('F1 Scores by Emotion Class')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save plot
        analysis_path = os.path.join(checkpoint_dir, "emotion_f1_analysis.png")
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Emotion F1 analysis saved to {analysis_path}")
        
        # Additional analyses can be added here

    def export_model_for_inference(self, model, checkpoint_dir):
        """
        Export model for inference with TFLite or other formats
        
        Args:
            model (tf.keras.Model): Trained model
            checkpoint_dir (str): Directory to save exported model
        """
        # Create export directory
        export_dir = os.path.join(checkpoint_dir, "export")
        os.makedirs(export_dir, exist_ok=True)
        
        # Save model in SavedModel format
        saved_model_path = os.path.join(export_dir, "saved_model")
        model.save(saved_model_path)
        logger.info(f"Saved model exported to {saved_model_path}")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = os.path.join(export_dir, "model.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite model exported to {tflite_path}")
        
        # Create a model info file
        model_info = {
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "num_layers": len(model.layers),
            "model_size_mb": os.path.getsize(tflite_path) / (1024 * 1024),
            "exported_timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(export_dir, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)

# Example usage
if __name__ == "__main__":
    # Check if running in a notebook
    try:
        # This will only exist in IPython/Jupyter environments
        get_ipython()
        IN_NOTEBOOK = True
    except NameError:
        IN_NOTEBOOK = False
    
    if IN_NOTEBOOK:
        # Use these default settings when in a notebook
        config_path = "config.yaml"
        epochs = 50
        fine_tune = True
        fine_tune_epochs = 20
        batch_size = None
        learning_rate = None
        model_type = "image"
        
        print("Running in notebook mode with default settings")
        
        # Initialize trainer
        trainer = DogEmotionTrainer(config_path=config_path)
        
        # Override batch size if provided
        if batch_size:
            trainer.config['model']['batch_size'] = batch_size
        
        # Train model
        model, history, checkpoint_dir = trainer.train_model(
            model_type=model_type,
            epochs=epochs,
            learning_rate=learning_rate,
            fine_tune=fine_tune,
            fine_tune_epochs=fine_tune_epochs
        )
        
        # The rest of your code...
        
    else:
        
        parser = argparse.ArgumentParser(description="Dog Emotion Trainer")
        # Parse command line arguments
        import argparse
        
        parser = argparse.ArgumentParser(description="Dog Emotion Trainer")
        parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
        parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
        parser.add_argument("--fine-tune", action="store_true", help="Enable fine-tuning after initial training")
        parser.add_argument("--fine-tune-epochs", type=int, default=20, help="Number of fine-tuning epochs")
        parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
        parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
        parser.add_argument("--model-type", type=str, default="image", choices=["image", "video"], 
                            help="Model type (image or video)")
    
        args = parser.parse_args()
    
    # Initialize trainer
    trainer = DogEmotionTrainer(config_path=args.config)
    
    # Override batch size if provided
    if args.batch_size:
        trainer.config['model']['batch_size'] = args.batch_size
    
    # Train model
    model, history, checkpoint_dir = trainer.train_model(
        model_type=args.model_type,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.fine_tune_epochs
    )
    
    # Plot training history
    trainer.plot_training_history(history, args.model_type, checkpoint_dir)
    
    # Evaluate model
    trainer.evaluate_model(model, args.model_type, checkpoint_dir)
    
    # Analyze model performance
    trainer.analyze_model_performance(checkpoint_dir)
    
    # Export model for inference
    trainer.export_model_for_inference(model, checkpoint_dir)
    
    logger.info("Training and evaluation completed successfully!")


