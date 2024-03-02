import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import os
import cv2

model = load_model(os.path.join("saved_models", "mnist", "best_model.keras"))

file = "2.png"
sample_path = "datasets/mnist/"+file

def pre_process(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(28,28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    return img

img = pre_process(sample_path)


# plt.imshow(img[0])  # Access the first element from the batch dimension
# plt.title("9")
# plt.axis("off")  # Hide unnecessary axis labels
# plt.show()

_prediction = model.predict(img)
print(_prediction)

print("file:", file)
print("digit : ", np.argmax(_prediction))
print("img.shape : ", img.shape)