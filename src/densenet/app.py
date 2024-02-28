import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import math

from _util import download_file

cwd = os.getcwd()

BASE_DIR = os.path.join(cwd, "datasets", "densenet")
ZIP_FILE_NAME = "alien-vs-predator-images.zip"
ZF_PATH = os.path.join(BASE_DIR, ZIP_FILE_NAME)

"""     Downloading Dataset     """

os.makedirs(BASE_DIR, exist_ok=True)
if not os.path.exists(ZF_PATH):
    os.system("kaggle datasets download -d pmigdal/alien-vs-predator-images -p ./datasets/densenet")

"""     EXTRACTING DATASET      """
zfile = zipfile.ZipFile(ZF_PATH)
zfile.extractall(BASE_DIR)

"""     Setting variables        """

base_dir = os.path.join(BASE_DIR, "alien_vs_predator_thumbnails","data")
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')

print(f"base_dir {os.path.isdir(base_dir)} ={base_dir}")
print(f"train_dir {os.path.isdir(train_dir)} ={train_dir}")
print(f"validation_dir {os.path.isdir(validation_dir)} ={validation_dir}")
def run():

    BATCH_SIZE = 8

    IMG_HEIGHT = 224
    IMG_WIDTH = 224

    epochs = 100

    """             Training            """

    """
    Preparing Images
    """

    train_dataset = image_dataset_from_directory(train_dir,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE,
                                                label_mode="binary",
                                                color_mode="rgb",
                                                image_size=(IMG_HEIGHT,IMG_WIDTH))

    validation_dataset = image_dataset_from_directory(validation_dir,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,
                                                    label_mode="binary",
                                                    color_mode="rgb",
                                                    image_size=(IMG_HEIGHT,IMG_WIDTH))

    class_names = train_dataset.class_names
    print(class_names)

    """## Visualizing"""

    # plt.figure(figsize=(16,9))
    # for images, labels in train_dataset.take(1):
    #   for i in range(BATCH_SIZE):
    #     plt.subplot(2,4,i+1)
    #     plt.imshow(images[i].numpy().astype("uint8"))
    #     plt.title(class_names[int(labels[i])])
    #     plt.axis("on")

    """## Image Augmentation"""

    data_augmentation = Sequential(
        [
            RandomFlip(input_shape=(IMG_HEIGHT,IMG_WIDTH,3), mode='horizontal'),
            RandomRotation(0.1),
            RandomZoom(0.1)
        ],
        name="Augmentation"
    )

    data_augmentation.summary()

    """## Test Dataset"""

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    print("No. of validation batches : %d" % tf.data.experimental.cardinality(validation_dataset))
    print("No. of test batches : %d"% tf.data.experimental.cardinality(test_dataset))

    """## Improvements in Dataset"""

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    """## Visualize Original and Augmented Dataset"""

    # plt.figure(figsize=(16,9))
    # for images, labels in train_dataset.take(1):
    #   argImg = data_augmentation(images)
    #   for i in range(int(BATCH_SIZE/2)):
    #     plt.subplot(2,4,i+1)
    #     plt.imshow(images[i].numpy().astype("uint8"))
    #     plt.title("Original")
    #     plt.axis("off")

    #     plt.subplot(2,4,i+1+int(BATCH_SIZE/2))
    #     plt.imshow(argImg[i].numpy().astype("uint8"))
    #     plt.title("Augmented")
    #     plt.axis("off")

    """
    DenseNet-201
    """

    base_densenet_model = Sequential(
        [
            Rescaling(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), scale=1.0 / 255, name="Rescaling"
            ),
            data_augmentation,
            DenseNet201(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                weights="imagenet",
                include_top=False,
            ),
        ],
        name="base_densenet_model",
    )

    base_densenet_model.trainable = False
    base_densenet_model.summary()

    custom_densenet_model = Sequential(
        [
            base_densenet_model,
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(units=1, activation="sigmoid"),
        ],
        name="custom_densenet_model",
    )

    custom_densenet_model.summary()

    custom_densenet_model.compile(
        optimizer=Adam(learning_rate=0.001*3),
        loss=BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    """## Fit the Model"""

    early = EarlyStopping(monitor="val_loss",
                        patience=math.floor(epochs*0.1))

    learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss",
                                                patience=2,
                                                verbose=1,
                                                factor=0.3,
                                                min_lr=0.000001)

    modelcheck = ModelCheckpoint(os.path.join('saved_models','best_model.hdf5'),
                                monitor='val_accuracy',
                                verbose=1,
                                save_best_only=True,
                                mode='max')

    history = custom_densenet_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs = epochs,
        callbacks = [modelcheck, learning_rate_reduction, early],
        verbose=1
    )

    """# Evaluate"""

    pd.DataFrame(custom_densenet_model.history.history).plot()

    test_accu = custom_densenet_model.evaluate(test_dataset)
    print("Testing Accuracy = ", test_accu[1]*100,"%")

run()