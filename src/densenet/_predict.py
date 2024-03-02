import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import os

IMG_HEIGHT = 224
IMG_WIDTH = 224

def pre_process(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    return img
 
def predictImg(img):
    model = tf.keras.models.load_model(os.path.join("saved_models","best_model.hdf5"))
    print(model.layers[-1].output_shape[-1])
    # return
    prediction = model.predict(new_image)


    print("Prediction", prediction)
    predicted_proba = prediction[0][0]  # Assuming binary classification
    print(f"Predicted probability of belonging to class 0 (alien): {predicted_proba}")
    print(f"Predicted probability of belonging to class 1 (predator): {1 - predicted_proba}")
    if predicted_proba > 0.5:
        print("Predicted class: Alien")
    else:
        print("Predicted class: Predator")

    

if __name__ == "__main__":

    image_type = random.choice(("alien", "predator"))
    dir_path = os.path.join(os.getcwd(), "datasets", "densenet", "alien_vs_predator_thumbnails", "data", "validation", image_type)
    img_path = os.path.join(dir_path, random.choice(os.listdir(dir_path)))
    
    new_image = pre_process(img_path)
    
    # plt.imshow(new_image[0])  # Access the first element from the batch dimension
    # plt.title(image_type)
    # plt.axis("off")  # Hide unnecessary axis labels
    # plt.show()
    
    predictImg(new_image)
    print(f"Original : {image_type}")