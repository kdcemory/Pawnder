# Quick test script
import tensorflow as tf

# Replace with your actual path
model_path = "C:\\Users\\kelly\\Documents\\GitHub\\Pawnder\\Models\\enhanced_dog_emotion_20250525-134150\\best_model.keras"
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")