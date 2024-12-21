import tensorflow as tf
import numpy as np
import json
import argparse
from tensorflow.keras.utils import load_img, img_to_array

# Define the preprocess_image function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Predict the class of a flower image.')
parser.add_argument('image_path', type=str, help='Path to the image file.')
parser.add_argument('model_path', type=str, help='Path to the trained model file.')
parser.add_argument('--top_k', type=int, default=5, help='Return the top K predictions.')
parser.add_argument('--category_names', type=str, default='label_map.json', help='Path to the label map JSON file.')
args = parser.parse_args()

# Load the model
model = tf.keras.models.load_model(args.model_path)

# Load the label map
with open(args.category_names, 'r') as f:
    class_names = json.load(f)

# Preprocess the input image
image = preprocess_image(args.image_path)

# Make predictions
predictions = model.predict(image)[0]

# Get the top K predictions
top_k_indices = np.argsort(predictions)[-args.top_k:][::-1]
top_k_classes = [class_names[str(i)] for i in top_k_indices]
top_k_probs = [predictions[i] for i in top_k_indices]

# Print the top K predictions
print("Top K Predictions:")
for i in range(len(top_k_classes)):
    print(f"{top_k_classes[i]}: {top_k_probs[i]:.2f}")
