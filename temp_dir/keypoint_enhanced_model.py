# keypoint_enhanced_model.py
# Enhanced model architecture that incorporates keypoint data

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, Flatten, Concatenate, BatchNormalization
)
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.optimizers import Adam
import yaml
import os

class KeypointEnhancedModel:
    """Model architecture for dog emotion recognition with keypoints"""
    
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
    
    def _create_image_branch(self, input_shape, backbone="mobilenetv2"):
        """
        Create the image processing branch of the model
        
        Args:
            input_shape (tuple): Image input shape (height, width, channels)
            backbone (str): Backbone model for transfer learning
            
        Returns:
            tuple: (input_tensor, output_tensor)
        """
        input_tensor = Input(shape=input_shape, name="image_input")
        
        # Select backbone based on configuration
        if backbone.lower() == "mobilenetv2":
            base_model = MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        elif backbone.lower() == "resnet50":
            base_model = ResNet50(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        elif backbone.lower() == "efficientnetb0":
            base_model = EfficientNetB0(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze early layers for transfer learning
        for layer in base_model.layers[:int(len(base_model.layers) * 0.7)]:
            layer.trainable = False
        
        # Connect input to backbone
        x = base_model(input_tensor)
        
        # Add custom layers on top of backbone
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config["model"]["dropout_rate"])(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config["model"]["dropout_rate"] / 2)(x)
        
        return input_tensor, x
    
    def _create_behavior_branch(self, behavior_size):
        """
        Create the behavioral indicator branch of the model
        
        Args:
            behavior_size (int): Number of behavioral indicator features
            
        Returns:
            tuple: (input_tensor, output_tensor)
        """
        input_tensor = Input(shape=(behavior_size,), name="behavior_input")
        
        x = Dense(128, activation="relu")(input_tensor)
        x = BatchNormalization()(x)
        x = Dropout(self.config["model"]["dropout_rate"] / 2)(x)
        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)
        
        return input_tensor, x
    
    def _create_keypoint_branch(self, num_keypoints=17):
        """
        Create the keypoint branch of the model
        
        Args:
            num_keypoints (int): Number of keypoints
            
        Returns:
            tuple: (input_tensor, keypoint_tensor, visibility_tensor, output_tensor)
        """
        # Main keypoint input - shape (num_keypoints, 2) for x,y coordinates
        keypoint_input = Input(shape=(num_keypoints, 2), name="keypoint_input")
        
        # Visibility input - shape (num_keypoints,) for visibility flags
        visibility_input = Input(shape=(num_keypoints,), name="keypoint_visibility")
        
        # Reshape visibility to match keypoints shape for multiplication
        visibility_reshaped = tf.reshape(visibility_input, (-1, num_keypoints, 1))
        
        # Apply visibility to keypoints (0 visibility will zero out the coordinate)
        masked_keypoints = keypoint_input * tf.concat([visibility_reshaped, visibility_reshaped], axis=2)
        
        # Flatten the keypoints
        x = Flatten()(masked_keypoints)
        
        # Add dense layers
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config["model"]["dropout_rate"] / 2)(x)
        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)
        
        return keypoint_input, visibility_input, x
    
    def create_model(self, model_type="image", num_emotions=14, behavior_size=64, num_keypoints=17, use_keypoints=True):
        """
        Create the full model architecture
        
        Args:
            model_type (str): 'image' or 'video'
            num_emotions (int): Number of emotion classes
            behavior_size (int): Number of behavioral indicator features
            num_keypoints (int): Number of keypoints
            use_keypoints (bool): Whether to include keypoint branch
            
        Returns:
            tf.keras.Model: Compiled model
        """
        # Get model configuration
        img_size = tuple(self.config["model"]["image_size"])
        dropout_rate = self.config["model"]["dropout_rate"]
        learning_rate = self.config["model"]["learning_rate"]
        backbone = self.config["model"]["backbone"]
        
        # Create branches
        image_input, image_features = self._create_image_branch(img_size, backbone)
        behavior_input, behavior_features = self._create_behavior_branch(behavior_size)
        
        # Combine features - with or without keypoints
        if use_keypoints:
            keypoint_input, visibility_input, keypoint_features = self._create_keypoint_branch(num_keypoints)
            combined = Concatenate()([image_features, behavior_features, keypoint_features])
        else:
            combined = Concatenate()([image_features, behavior_features])
        
        # Add fusion layers
        x = Dense(256, activation="relu")(combined)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate / 4)(x)
        
        # Create output heads
        emotion_output = Dense(num_emotions, activation="softmax", name="emotion_output")(x)
        confidence_output = Dense(1, activation="sigmoid", name="confidence_output")(x)
        
        # Create model with appropriate inputs
        if use_keypoints:
            model = Model(
                inputs=[image_input, behavior_input, keypoint_input, visibility_input],
                outputs=[emotion_output, confidence_output]
            )
        else:
            model = Model(
                inputs=[image_input, behavior_input],
                outputs=[emotion_output, confidence_output]
            )
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                "emotion_output": "categorical_crossentropy",
                "confidence_output": "binary_crossentropy"
            },
            metrics={
                "emotion_output": ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
                "confidence_output": ["accuracy"]
            },
            loss_weights={
                "emotion_output": 1.0,
                "confidence_output": 0.2
            }
        )
        
        return model
    
    def fine_tune_model(self, model, num_layers=15):
        """
        Fine-tune the model by unfreezing backbone layers
        
        Args:
            model (tf.keras.Model): Trained model
            num_layers (int): Number of layers to unfreeze from the end
            
        Returns:
            tf.keras.Model: Model ready for fine-tuning
        """
        # Find the backbone model (MobileNetV2, etc.)
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                backbone = layer
                break
        else:
            # If no backbone model is found
            return model
        
        # Unfreeze the last n layers
        for layer in backbone.layers[-(num_layers):]:
            layer.trainable = True
        
        # Use a lower learning rate for fine-tuning
        lr = self.config["model"]["learning_rate"] / 10
        
        # Recompile the model
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss={
                "emotion_output": "categorical_crossentropy",
                "confidence_output": "binary_crossentropy"
            },
            metrics={
                "emotion_output": ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
                "confidence_output": ["accuracy"]
            },
            loss_weights={
                "emotion_output": 1.0,
                "confidence_output": 0.2
            }
        )
        
        return model