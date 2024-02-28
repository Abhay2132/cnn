import numpy as np
import tensorflow as tf
import cv2
import os

IMG_HEIGHT = 224
IMG_WIDTH = 224

img_path = os.path.join(os.getcwd(), "datasets", "densenet", "alien_vs_predator_thumbnails", "data", "validation", "alien", "50.jpg")
new_image = cv2.imread(img_path)

# Assuming your model expects IMG_HEIGHT and IMG_WIDTH as defined earlier
new_image = cv2.resize(new_image, (IMG_WIDTH, IMG_HEIGHT))

if cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB).shape[-1] == 3:
    # Image is already RGB
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
else:
    # Convert from BGR to RGB if needed
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

new_image = new_image.astype('float32') / 255.0

# Assuming you have your preprocessed new image as `new_image` (a NumPy array):
new_image = np.expand_dims(new_image, axis=0)  # Add a batch dimension

import matplotlib.pyplot as plt

# Assuming you have already preprocessed `new_image` for prediction
plt.imshow(new_image[0])  # Access the first element from the batch dimension
plt.title("Preprocessed Image")
plt.axis("off")  # Hide unnecessary axis labels
plt.show()

# Load the model
model = tf.keras.models.load_model("best_model.hdf5")

# Predict on the new image
prediction = model.predict(new_image)

print("Prediction", prediction)

# Access the predicted probability
predicted_proba = prediction[0][0]  # Assuming binary classification

print(f"Predicted probability of belonging to class 0 (alien): {predicted_proba}")
print(f"Predicted probability of belonging to class 1 (predator): {1 - predicted_proba}")

# Use a threshold (e.g., 0.5) to classify based on probability if needed
if predicted_proba > 0.5:
    print("Predicted class: Alien")
else:
    print("Predicted class: Predator")
