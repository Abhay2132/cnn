import tensorflow as tf
from PIL import Image
import numpy as np
from random import choice
import os

files_limit = 20
categories = ("alien", "predator")
files = list(range(1,files_limit))
files = tuple(map(lambda _ : str(_)+".jpg", files))

dir_path = "datasets/densenet/alien_vs_predator_thumbnails/data/train/"
saved_model_path = "saved_models/best_model.hdf5"
# saved_model_path = "saved_models/densenet/alien-predator.keras"

model = tf.keras.models.load_model(saved_model_path)

print(files)
# raise Exception("")

def choiceRandomImage(category, file):    
    # category = choice(categories)
    imgs_dir = os.path.join(dir_path, category)
    # file = choice(os.listdir(imgs_dir))
    img_path = os.path.join(dir_path, category, file)
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    return (img_array, category, file)


def predict(data):
    (img_array, category, file) = data
    prediction = model.predict(img_array)
    predicted_value = prediction[0][0]
    prediction = model.predict(img_array)
    predicted_value = prediction[0][0]

    predicted = ""
    if predicted_value > 0.5:
        print("Predicted class: Alien")
        predicted = "Alien"
    else:
        print("Predicted class: Predator")
        predicted = "Predator"

    print("original : ", category)
    print("file : " , file)
    print("predicted value : ", predicted_value)

    return (predicted, category, predicted_value)

os.system("cls")

original = []
predicted = []
probabilities = []
for i in range(1, 19):
    print("\n")
    (p,o,pv) = predict(choiceRandomImage(categories[i%2], files[i]))
    original.append(o)
    predicted.append(p)
    probabilities.append(pv)

print("Original    Predicted    Probability")
for (o,p,P) in zip(original, predicted, probabilities):
    print(f"{o}{' '*(10-len(o))}    {p}{(' '*(10-len(p)))}    {P}")



