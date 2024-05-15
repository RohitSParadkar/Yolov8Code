import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from io import BytesIO

def predict_defect(image_path, model_path):
  """
  Predicts defects in an image using a TensorFlow (.h5) model.

  Args:
      image_path (str): Path to the image file.
      model_path (str): Path to the TensorFlow (.h5) model file.

  Returns:
      PIL.Image: The image with predicted defects visualized (optional).
      np.ndarray: The predicted class probabilities (optional).
  """

  # Load the image
  img = load_img(image_path, target_size=(224, 224))  # Adjust based on your model's input size

  # Preprocess the image (assuming model expects normalized input)
  x = img_to_array(img)
  x = x / 255.0
  x = np.expand_dims(x, axis=0)  # Add batch dimension

  # Load the TensorFlow model
  model = tf.keras.models.load_model(model_path)

  # Make prediction
  predictions = model.predict(x)

  # Visualize predictions (optional, replace with your defect visualization logic)
  # This example demonstrates basic coloring based on predicted class
  if predictions.shape[1] > 1:  # Multi-class classification
    predicted_class = np.argmax(predictions, axis=1)[0]
    colors = [(0, 255, 0), (255, 0, 0)]  # Green for defect, red for non-defect
    img_array = np.asarray(img)
    img_array[..., 1] = np.where(predicted_class == 0, img_array[..., 1], colors[0][1])
    img_array[..., 2] = np.where(predicted_class == 0, img_array[..., 2], colors[0][2])
    img = Image.fromarray(img_array, 'RGB')
  else:  # Binary classification (defect vs. non-defect)
    predicted_prob = predictions[0][0]
    threshold = 0.5  # Adjust threshold based on your model's behavior
    if predicted_prob > threshold:
      img = img.convert('L')  # Convert to grayscale for highlighting
      img = img.point(lambda p: 0 if p > 128 else 255)  # Threshold and invert
    else:
      pass  # No visual change for predicted non-defect

  return img, predictions

def display_image(image):
  """
  Displays the image using PyQt.

  Args:
      image (PIL.Image): The image to display.
  """

  image_bytes = BytesIO()
  image.save(image_bytes, format='PNG')
  image_bytes.seek(0)

  app = QApplication([])
  widget = QWidget()
  layout = QVBoxLayout()
  label = QLabel()
  pixmap = QPixmap()
  pixmap.loadFromData(image_bytes.getvalue())
  label.setPixmap(pixmap)
  layout.addWidget(label)
  widget.setLayout(layout)

  window_width = image.width + 10
  window_height = image.height + 10
  widget.setGeometry(10, 10, window_width, window_height)

  widget.setWindowTitle("Defect Prediction Result")
  widget.show()
  app.exec_()

# Example usage
image_path = "../output/testImage/test2.jpg"  # Replace with your image path
model_path = "../models/model.h5"  # Replace with your model path

predicted_image, predictions = predict_defect(image_path, model_path)
display_image(predicted_image)

# Access predictions array for further analysis (e.g., class probabilities)
print(predictions)
