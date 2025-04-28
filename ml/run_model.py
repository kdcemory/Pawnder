"""
Updated Dog Emotion Model Execution Script

This script makes it easy to run the improved dog emotion classification model
that handles class name inconsistencies and uses the correct data paths.
"""

import os
import sys
import traceback
from datetime import datetime

def create_necessary_files():
    """Create necessary files if they don't exist"""
    # Check if improved model file exists
    if not os.path.exists('dog_emotion_with_behaviors_fixed.py'):
        print("Error: dog_emotion_with_behaviors_fixed.py is missing.")
        print("Please create this file with the provided code.")
        return False
    
    return True

def check_environment():
    """Check if the environment is properly set up"""
    try:
        # Check for required Python packages
        import tensorflow as tf
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import cv2
        
        # Check TensorFlow version
        tf_version = tf.__version__
        print(f"TensorFlow version: {tf_version}")
        
        # Check for GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"Found {len(physical_devices)} GPU(s):")
            for i, device in enumerate(physical_devices):
                print(f"  [{i}] {device}")
            
            # Try to enable memory growth
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Memory growth enabled for all GPUs")
            except Exception as e:
                print(f"Could not enable memory growth: {str(e)}")
        else:
            print("No GPUs found. Training will use CPU only.")
        
        # Verify data paths
        base_dir = "/content/drive/MyDrive/Colab Notebooks/Pawnder"
        processed_dir = os.path.join(base_dir, "Data/processed")
        model_dir = os.path.join(base_dir, "Models")
        
        print("\nVerifying data paths:")
        print(f"Base directory exists: {os.path.exists(base_dir)}")
        print(f"Processed directory exists: {os.path.exists(processed_dir)}")
        print(f"Model directory exists: {os.path.exists(model_dir)}")
        
        # Check for annotations files
        train_annotations = os.path.join(processed_dir, "train/annotations/annotations.json")
        val_annotations = os.path.join(processed_dir, "validation/annotations/annotations.json")
        test_annotations = os.path.join(processed_dir, "test/annotations/annotations.json")
        
        print(f"Train annotations exist: {os.path.exists(train_annotations)}")
        print(f"Validation annotations exist: {os.path.exists(val_annotations)}")
        print(f"Test annotations exist: {os.path.exists(test_annotations)}")
        
        return True
    except ImportError as e:
        print(f"Missing required package: {str(e)}")
        return False
    except Exception as e:
        print(f"Environment check failed: {str(e)}")
        return False

def train_model(epochs=50, batch_size=32, fine_tune=True):
    """Train the model with given parameters"""
    print("\n" + "="*80)
    print("Training dog emotion model...")
    print("="*80)
    
    try:
        # Import the model
        from dog_emotion_with_behaviors_fixed import DogEmotionWithBehaviors
        
        # Create model instance
        classifier = DogEmotionWithBehaviors()
        
        # Train model
        history, model_dir = classifier.train(
            epochs=epochs,
            batch_size=batch_size,
            fine_tune=fine_tune
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {model_dir}")
        
        return True, model_dir
    except ImportError:
        print("Could not import dog_emotion_with_behaviors_fixed.py.")
        print("Make sure dog_emotion_with_behaviors_fixed.py is in the current directory.")
        return False, None
    except Exception as e:
        print(f"Training failed: {str(e)}")
        traceback.print_exc()
        return False, None

def main():
    """Main function to run the improved model"""
    print("="*80)
    print("Improved Dog Emotion Classifier Execution")
    print("="*80)
    
    # Check if necessary files exist
    if not create_necessary_files():
        return
    
    # Check environment
    print("\nChecking environment...")
    env_ok = check_environment()
    if not env_ok:
        print("Environment check failed. Please ensure all required packages are installed.")
        return
    
    # Configure training parameters
    print("\nTraining configuration:")
    epochs = 50
    batch_size = 32
    fine_tune = True
    
    # Ask for confirmation before training
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Fine-tuning: {'Enabled' if fine_tune else 'Disabled'}")
    
    choice = input("\nStart training with these parameters? (y/n): ")
    if choice.lower() != 'y':
        print("Training canceled.")
        return
    
    # Start timer
    start_time = datetime.now()
    print(f"\nStarting training at {start_time.strftime('%Y-%m-%d %H:%M:%S')}...")
    
    # Train model
    success, model_dir = train_model(
        epochs=epochs,
        batch_size=batch_size,
        fine_tune=fine_tune
    )
    
    # Calculate training time
    end_time = datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if success:
        print("\n" + "="*80)
        print(f"Training completed in {hours}h {minutes}m {seconds}s")
        print(f"Model saved to: {model_dir}")
        print("="*80)
    else:
        print("\n" + "="*80)
        print(f"Training failed after {hours}h {minutes}m {seconds}s")
        print("Please check the error messages above.")
        print("="*80)

if __name__ == "__main__":
    main()