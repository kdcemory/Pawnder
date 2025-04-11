# dog_behavior_model_training.py
# Training script for Pawnder dog behavior analysis model

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import time
import datetime
from google.colab import drive

# Import project configuration
from config import DATA_DIRS, EMOTION_MAPPING, ensure_directories, setup_aws_from_secrets
from s3_sync import S3Sync

# Configure paths and settings
MODEL_NAME = "pawnder_behavior_v1"
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 100
LEARNING_RATE = 0.0001
PATIENCE = 10  # For early stopping
CLASS_NAMES = ["Happy/Playful", "Relaxed", "Submissive/Appeasement", 
               "Curiosity/Alertness", "Stressed", "Fearful/Anxious", 
               "Aggressive/Threatening"]

# Setup AWS connectivity
setup_aws_from_secrets()
s3_sync = S3Sync(bucket_name='pawnder-media-storage')

def create_model_directories():
    """Create directories for model outputs"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create local model directory
    model_dir = os.path.join(DATA_DIRS['models'], MODEL_NAME, timestamp)
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    plots_dir = os.path.join(model_dir, 'plots')
    results_dir = os.path.join(model_dir, 'results')
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    return {
        'base': model_dir,
        'checkpoints': checkpoints_dir,
        'plots': plots_dir,
        'results': results_dir,
        'timestamp': timestamp
    }

def prepare_datasets():
    """Prepare training, validation and test datasets"""
    # Sync data from S3 if configured
    if 'aws_data' in DATA_DIRS:
        print("Syncing data from S3...")
        s3_sync.download_folder('training', DATA_DIRS['processed'])
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load datasets
    train_dir = os.path.join(DATA_DIRS['processed'], 'train', 'images')
    valid_dir = os.path.join(DATA_DIRS['processed'], 'validation', 'images')
    test_dir = os.path.join(DATA_DIRS['processed'], 'test', 'images')
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True
    )
    
    validation_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )
    
    return {
        'train': train_generator,
        'validation': validation_generator,
        'test': test_generator
    }

def build_model(num_classes):
    """Build model architecture using transfer learning"""
    # Use MobileNetV2 as base model (efficient for mobile deployment)
    base_model = applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Build model on top of the base model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, datasets, model_dirs):
    """Train the model with early stopping and checkpoints"""
    # Setup callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dirs['checkpoints'], 'model-{epoch:02d}-{val_accuracy:.4f}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(model_dirs['base'], 'logs')
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Train the model
    history = model.fit(
        datasets['train'],
        epochs=EPOCHS,
        validation_data=datasets['validation'],
        callbacks=callbacks_list
    )
    
    # Save the final model
    model.save(os.path.join(model_dirs['base'], f'{MODEL_NAME}_final.h5'))
    
    return history

def plot_training_history(history, model_dirs):
    """Plot and save training history"""
    # Accuracy plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dirs['plots'], 'training_history.png'))
    
    # Save history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(model_dirs['results'], 'training_history.csv'), index=False)

def evaluate_model(model, datasets, model_dirs):
    """Evaluate model on test set and save results"""
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(datasets['test'])
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Get predictions
    predictions = model.predict(datasets['test'])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = datasets['test'].classes
    
    # Calculate classification report
    class_names = list(datasets['test'].class_indices.keys())
    report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(model_dirs['results'], 'classification_report.csv'))
    
    # Create and save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_classes, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dirs['plots'], 'confusion_matrix.png'))
    
    # Save evaluation results
    eval_results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'classification_report': report
    }
    
    with open(os.path.join(model_dirs['results'], 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    return eval_results

def fine_tune_model(model, datasets, model_dirs):
    """Fine-tune the model by unfreezing some layers"""
    # Unfreeze the top layers of the base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks for fine-tuning
    fine_tune_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dirs['checkpoints'], 'fine_tuned-{epoch:02d}-{val_accuracy:.4f}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Fine-tune model
    fine_tune_history = model.fit(
        datasets['train'],
        epochs=20,
        validation_data=datasets['validation'],
        callbacks=fine_tune_callbacks
    )
    
    # Save fine-tuned model
    model.save(os.path.join(model_dirs['base'], f'{MODEL_NAME}_fine_tuned.h5'))
    
    return fine_tune_history

def sync_results_to_s3(model_dirs):
    """Sync model results to S3"""
    if 'aws_models' in DATA_DIRS:
        print("Syncing model results to S3...")
        s3_prefix = f"models/{MODEL_NAME}/{model_dirs['timestamp']}"
        s3_sync.upload_folder(model_dirs['base'], s3_prefix)
        print(f"✅ Model results uploaded to S3: {s3_prefix}")

def main():
    """Main training function"""
    print("Starting Pawnder behavior model training...")
    
    # Setup model directories
    model_dirs = create_model_directories()
    print(f"Model outputs will be saved to: {model_dirs['base']}")
    
    # Prepare datasets
    print("Preparing datasets...")
    datasets = prepare_datasets()
    print(f"Loaded {datasets['train'].samples} training samples")
    print(f"Loaded {datasets['validation'].samples} validation samples")
    print(f"Loaded {datasets['test'].samples} test samples")
    
    # Build model
    print("Building model...")
    model = build_model(len(CLASS_NAMES))
    model.summary()
    
    # Train model
    print("Training model...")
    history = train_model(model, datasets, model_dirs)
    
    # Plot training results
    print("Generating training plots...")
    plot_training_history(history, model_dirs)
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = evaluate_model(model, datasets, model_dirs)
    
    # Fine-tune model if initial accuracy is reasonable
    if eval_results['test_accuracy'] > 0.6:
        print("Fine-tuning model...")
        fine_tune_history = fine_tune_model(model, datasets, model_dirs)
        plot_training_history(fine_tune_history, model_dirs)
        # Re-evaluate fine-tuned model
        print("Evaluating fine-tuned model...")
        eval_results = evaluate_model(model, datasets, model_dirs)
    
    # Sync results to S3
    sync_results_to_s3(model_dirs)
    
    print("Model training complete!")
    return model, eval_results, model_dirs

if __name__ == "__main__":
    # Mount Google Drive if running in Colab
    try:
        drive.mount('/content/drive', force_remount=True)
        print("Google Drive mounted successfully")
    except:
        print("Not running in Colab or drive already mounted")
    
    # Run the training pipeline
    model, eval_results, model_dirs = main()
