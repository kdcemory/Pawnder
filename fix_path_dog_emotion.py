Dog Emotion Model with Behavioral Feature Integration - Fixed Path Handling

This implementation focuses on integrating the Primary Behavior Matrix
features while maintaining compatibility with TensorFlow's recent versions.
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
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

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

class DogEmotionWithBehaviors:
    """
    Dog emotion classification model with behavioral feature integration
    """
    
    def __init__(self, base_dir="/content/drive/MyDrive/Colab Notebooks/Pawnder"):
        """
        Initialize the classifier
        
        Args:
            base_dir: Base directory containing the project
        """
        self.base_dir = base_dir
        self.processed_dir = os.path.join(base_dir, "Data/processed")
        self.model = None
        self.class_names = []
        self.behavior_columns = []
        self.model_dir = os.path.join(base_dir, "Models")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Print paths for debugging
        print(f"Base directory: {self.base_dir}")
        print(f"Processed directory: {self.processed_dir}")
        print(f"Model directory: {self.model_dir}")
        
        # Verify directories
        if not os.path.exists(self.processed_dir):
            print(f"Warning: Processed directory not found at {self.processed_dir}")
        
        if not os.path.exists(os.path.join(self.processed_dir, 'train_by_class')):
            print(f"Warning: Train directory not found at {os.path.join(self.processed_dir, 'train_by_class')}")
            # Try to look for alternative locations
            possible_dirs = [
                os.path.join(base_dir, "data/processed/train_by_class"),
                os.path.join(base_dir, "Data/processed/train_by_class"),
                os.path.join("/content/drive/MyDrive/Colab Notebooks/Pawnder/Data/processed/train_by_class")
            ]
            for possible_dir in possible_dirs:
                if os.path.exists(possible_dir):
                    print(f"Found alternative train directory: {possible_dir}")
                    # Update processed dir accordingly
                    self.processed_dir = os.path.dirname(possible_dir)
                    print(f"Updated processed directory to: {self.processed_dir}")
                    break
    
    def create_model(self, num_classes, behavior_size=10, img_size=(224, 224, 3)):
        """
        Create model with both image and behavioral inputs
        
        Args:
            num_classes: Number of emotion classes
            behavior_size: Number of behavioral features
            img_size: Input image size (height, width, channels)
            
        Returns:
            Compiled Keras model
        """
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
        output = layers.Dense(num_classes, activation='softmax')(combined)
        
        # Create and compile model
        model = tf.keras.Model(
            inputs={'image_input': image_input, 'behavior_input': behavior_input},
            outputs=output
        )
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
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
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model fine-tuned: Unfroze last {unfreeze_layers} layers of the base model")
        return self.model
    
    def load_class_data(self, split_name='train'):
        """
        Load image paths and labels from class directories
        
        Args:
            split_name: Data split ('train', 'validation', 'test')
            
        Returns:
            tuple: (image_paths, labels, class_names)
        """
        # Get split directory
        if split_name == 'train':
            split_dir = os.path.join(self.processed_dir, 'train_by_class')
        elif split_name == 'validation':
            split_dir = os.path.join(self.processed_dir, 'val_by_class')
        elif split_name == 'test':
            split_dir = os.path.join(self.processed_dir, 'test_by_class')
        else:
            raise ValueError(f"Invalid split name: {split_name}")
        
        # Check if directory exists
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Get class directories
        class_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d)) and d != 'unknown']
        
        # Sort for consistent order
        class_dirs.sort()
        
        # Save class names
        self.class_names = class_dirs
        
        # Prepare lists for images and labels
        image_paths = []
        labels = []
        absolute_paths = []  # Store absolute paths for debugging
        
        # Load images and labels
        print(f"Loading {split_name} data from {split_dir}")
        for class_idx, class_name in enumerate(class_dirs):
            class_dir = os.path.join(split_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"  {class_name}: {len(image_files)} images")
            
            # Add images and labels
            for img_file in image_files:
                # Store only the full absolute path
                abs_path = os.path.abspath(os.path.join(class_dir, img_file))
                image_paths.append(abs_path)
                absolute_paths.append(abs_path)
                labels.append(class_idx)
        
        # Save a few paths for debugging
        if len(absolute_paths) > 0:
            print(f"Example paths:")
            for i in range(min(3, len(absolute_paths))):
                print(f"  {absolute_paths[i]}")
                # Check if file exists
                print(f"    Exists: {os.path.exists(absolute_paths[i])}")
        
        return image_paths, labels, class_dirs
    
    def load_behavior_data(self):
        """
        Load behavioral data from Primary Behavior Matrix
        
        Returns:
            tuple: (behavior_data, behavior_columns)
        """
        # Try to find the primary matrix data
        matrix_paths = [
            os.path.join(self.base_dir, "Data/Matrix/primary_behavior_matrix.json"),
            os.path.join(self.base_dir, "Data/Matrix/Primary Behavior Matrix.xlsx"),
            os.path.join(self.base_dir, "data/Matrix/primary_behavior_matrix.json")
        ]
        
        behavior_data = {}
        behavior_columns = []
        
        # Try to load the matrix from any of the possible paths
        loaded = False
        
        # First try to load from annotations
        annotations_path = os.path.join(self.processed_dir, "combined_annotations.json")
        if os.path.exists(annotations_path):
            try:
                print(f"Loading annotations from {annotations_path}")
                with open(annotations_path, 'r') as f:
                    annotations = json.load(f)
                
                # Extract behavior columns
                first_item = next(iter(annotations.values()))
                behavior_columns = [k for k in first_item.keys() if k.startswith('behavior_')]
                
                if behavior_columns:
                    print(f"Found {len(behavior_columns)} behavior columns in annotations")
                    
                    # Create behavior data dictionary
                    for image_id, data in annotations.items():
                        # Extract behaviors
                        behaviors = []
                        for col in behavior_columns:
                            if col in data:
                                value = data[col]
                                # Convert to float
                                if isinstance(value, bool):
                                    value = 1.0 if value else 0.0
                                elif isinstance(value, (int, float)):
                                    value = float(value)
                                else:
                                    value = 0.0
                                behaviors.append(value)
                            else:
                                behaviors.append(0.0)
                        
                        # Store behaviors by full path and basename for easier matching
                        behavior_data[image_id] = behaviors
                        # Also store by basename for easier matching
                        basename = os.path.basename(image_id)
                        if basename:
                            behavior_data[basename] = behaviors
                    
                    loaded = True
                    print(f"Loaded behavior data for {len(behavior_data)} images")
            except Exception as e:
                print(f"Error loading behaviors from annotations: {str(e)}")
        
        # Try to load the Primary Behavior Matrix directly
        if not loaded:
            for matrix_path in matrix_paths:
                if os.path.exists(matrix_path):
                    try:
                        print(f"Trying to load matrix from {matrix_path}")
                        # Handle JSON format
                        if matrix_path.endswith('.json'):
                            with open(matrix_path, 'r') as f:
                                matrix_data = json.load(f)
                            
                            # Extract behavior columns
                            behavior_columns = [k for k in matrix_data.keys() if k.startswith('behavior_')]
                            
                            if behavior_columns:
                                print(f"Found {len(behavior_columns)} behavior columns in matrix")
                                loaded = True
                                break
                        
                        # Handle Excel format (requires pandas)
                        elif matrix_path.endswith('.xlsx'):
                            import pandas as pd
                            matrix_df = pd.read_excel(matrix_path)
                            
                            # Extract behavior columns
                            behavior_columns = [col for col in matrix_df.columns if col.startswith('behavior_')]
                            
                            if behavior_columns:
                                print(f"Found {len(behavior_columns)} behavior columns in Excel matrix")
                                loaded = True
                                break
                    except Exception as e:
                        print(f"Error loading matrix from {matrix_path}: {str(e)}")
        
        # If no behavior data found, create dummy data
        if not behavior_data:
            print("No behavior data found. Creating placeholder behavior data.")
            behavior_columns = ["behavior_dummy"]
            behavior_data = {}  # Will be filled with zeros during batching
        
        self.behavior_columns = behavior_columns
        return behavior_data, behavior_columns
    
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
            behavior_size = len(self.behavior_columns) if self.behavior_columns else 1
        
        # Create data generator class
        class DataGenerator(tf.keras.utils.Sequence):
            def __init__(self, image_paths, labels, behavior_data, behavior_size, img_size, batch_size, augment):
                self.image_paths = image_paths
                self.labels = labels
                self.behavior_data = behavior_data
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
                
                # Fill batch data
                for i, idx in enumerate(batch_indices):
                    # Get image path - already absolute from load_class_data
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
                        if self.behavior_data:
                            # Try a few different ways to match the behavior data
                            matched = False
                            
                            # 1. Try by full path
                            if img_path in self.behavior_data:
                                batch_behaviors[i] = self.behavior_data[img_path]
                                matched = True
                            
                            # 2. Try by basename
                            if not matched:
                                basename = os.path.basename(img_path)
                                if basename in self.behavior_data:
                                    batch_behaviors[i] = self.behavior_data[basename]
                                    matched = True
                            
                            # 3. Try by basename without extension
                            if not matched:
                                basename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
                                if basename_no_ext in self.behavior_data:
                                    batch_behaviors[i] = self.behavior_data[basename_no_ext]
                                    matched = True
                            
                            # If still no match, leave as zeros (already initialized)
                    
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
                        # Use zeros for failed images (already initialized)
                
                # Return the batch (using the proper input format for model)
                inputs = {
                    'image_input': batch_images,
                    'behavior_input': batch_behaviors
                }
                
                return inputs, batch_labels
            
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
        # Load data
        train_paths, train_labels, class_names = self.load_class_data('train')
        val_paths, val_labels, _ = self.load_class_data('validation')
        
        # Load behavior data
        behavior_data, behavior_columns = self.load_behavior_data()
        
        # Create data generators
        train_gen = self.create_data_generator(
            train_paths, train_labels, behavior_data, img_size, batch_size, augment=True)
        
        val_gen = self.create_data_generator(
            val_paths, val_labels, behavior_data, img_size, batch_size, augment=False)
        
        # Create model if not already created
        if self.model is None:
            self.create_model(
                num_classes=len(class_names),
                behavior_size=len(behavior_columns),
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
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            # Model checkpoints
            ModelCheckpoint(
                os.path.join(model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode='max'
            )
        ]
        
        # Train model
        print(f"Training model with {len(train_paths)} images and {len(behavior_columns)} behavior features")
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
                'behavior_columns': behavior_columns,
                'img_size': img_size,
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
        test_paths, test_labels, _ = self.load_class_data('test')
        
        # Load behavior data
        behavior_data, _ = self.load_behavior_data()
        
        # Create test generator
        test_gen = self.create_data_generator(
            test_paths, test_labels, behavior_data, (224, 224), batch_size=32, augment=False)
        
        # Evaluate model
        print("Evaluating model on test set...")
        evaluation = self.model.evaluate(test_gen)
        
        # Save evaluation metrics
        metrics = {
            'loss': float(evaluation[0]),
            'accuracy': float(evaluation[1])
        }
        
        with open(os.path.join(model_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate predictions
        print("Generating predictions for confusion matrix...")
        y_true = []
        y_pred = []
        
        for i in range(len(test_gen)):
            # Get batch data
            inputs, batch_y = test_gen[i]
            
            # Predict batch
            batch_pred = self.model.predict(inputs)
            
            # Get true and predicted classes
            batch_y_true = np.argmax(batch_y, axis=1)
            batch_y_pred = np.argmax(batch_pred, axis=1)
            
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
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print("Classification report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
    
    def plot_training_history(self, history, output_dir):
        """
        Plot training history
        
        Args:
            history: Training history
            output_dir: Directory to save plots
        """
        # Plot accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'accuracy.png'), dpi=300)
        plt.close()
        
        # Plot loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss.png'), dpi=300)
        plt.close()
    
    def predict_image(self, image_path, behavior_features=None):
        """
        Predict emotion for a single image
        
        Args:
            image_path: Path to image file
            behavior_features: Optional list of behavior features
            
        Returns:
            tuple: (predicted_class, confidence, all_predictions)
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
                behavior_size = len(self.behavior_columns) if self.behavior_columns else 1
                behavior_features = np.zeros((1, behavior_size), dtype=np.float32)
            else:
                behavior_features = np.array([behavior_features], dtype=np.float32)
            
            # Make prediction
            inputs = {
                'image_input': img,
                'behavior_input': behavior_features
            }
            predictions = self.model.predict(inputs)
            
            # Get predicted class and confidence
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            if len(self.class_names) > predicted_idx:
                predicted_class = self.class_names[predicted_idx]
            else:
                predicted_class = f"Class {predicted_idx}"
            
            # Create all predictions dict
            all_predictions = {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
            
            return predicted_class, confidence, all_predictions
        
        except Exception as e:
            print(f"Error predicting image: {str(e)}")
            return None, 0.0, {}
    
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
            
            # Load metadata if provided
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Extract metadata
                self.class_names = metadata.get('class_names', [])
                self.behavior_columns = metadata.get('behavior_columns', [])
                
                print(f"Loaded metadata: {len(self.class_names)} classes, {len(self.behavior_columns)} behavior features")
            
            return self.model
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

# Helper function to find the train_by_class directory
def find_directory(base_dir="/content/drive/MyDrive/Colab Notebooks/Pawnder", target_dir="train_by_class"):
    """
    Find a directory recursively starting from the base_dir
    
    Args:
        base_dir: Directory to start the search
        target_dir: Directory name to find
        
    Returns:
        Path to the found directory or None
    """
    print(f"Searching for {target_dir} in {base_dir}")
    
    for root, dirs, files in os.walk(base_dir):
        if target_dir in dirs:
            found_dir = os.path.join(root, target_dir)
            print(f"Found {target_dir} at {found_dir}")
            return found_dir
    
    return None

# Usage example
if __name__ == "__main__":
    # Try to find the correct directory
    train_dir = find_directory()
    if train_dir:
        # Use the parent directory as the processed_dir
        processed_dir = os.path.dirname(train_dir)
        print(f"Using {processed_dir} as the processed directory")
        
        # Create classifier with the found directory
        classifier = DogEmotionWithBehaviors(base_dir=os.path.dirname(processed_dir))
    else:
        # Use default directory
        classifier = DogEmotionWithBehaviors()
    
    # Train model
    history, model_dir = classifier.train(
        epochs=50,
        batch_size=32,
        fine_tune=True
    )
    
    print(f"Training completed. Model saved to {model_dir}")
    f.write("""
# Code content is in the artifact above
    """)