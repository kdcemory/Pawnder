# Convert your trained model to TensorFlow Serving format
# From your model directory
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('"C:\Users\kelly\Documents\GitHub\Pawnder\Models\enhanced_dog_emotion_20250525-134150\best_model.keras"')
tf.saved_model.save(model, 'exported_model/1')
"