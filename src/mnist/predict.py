import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

file = "2.png"
model = load_model(os.path.join("saved_models", "mnist", "best_model.keras"))

sample_path = "datasets/mnist/"+file

def preprocess_image(img_path):
  img = image.load_img(img_path, target_size=(28, 28), grayscale=True)  # Ensure grayscale
  img_array = image.img_to_array(img)
  img_array = img_array / 255.0
  img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
  return img_array


img = preprocess_image(sample_path)
print("img shape : ", img.shape)

plt.imshow(img[0], cmap='gray')  # Access the first element from the batch dimension
plt.title(file)
plt.axis("off")  # Hide unnecessary axis labels
plt.show()

_prediction = model.predict(img)

digit = np.argmax(_prediction)

print("File : ", file)
print("digit : ", digit)
print("img.shape : ", img.shape)