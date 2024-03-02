import tensorflow as tf
import tensorflow.keras as K

num_classes = 10

def preprocess_data(X,Y):
    Y_p = K.utils.to_categorical(Y, num_classes)
    X_p = K.application.densenet.preprocess_input(X)
    return X_p, Y_p

(X_train, Y_train),(X_test, y_test) = K.datasets.cifar10.load_data()

# load the Cifar10 dataset, 50,000 training images and 10,000 test images (here used as validation data)
(X, Y), (x_test, y_test) = K.datasets.cifar10.load_data()

# preprocess the data using the application's preprocess_input method and convert the labels to one-hot encodings
X_p, Y_p = preprocess_data(X, Y)
x_t, y_t = preprocess_data(x_test, y_test)

resized_images = K.layers.Lambda(lambda image: tf.image.resize(image, (224,224)))(input_tensor)