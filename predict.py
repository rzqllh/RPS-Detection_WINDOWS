import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from main import target_size, plt

# Define a function for making predictions on new images


def predict_image(path_to_image):
    img = image.load_img(path_to_image, target_size=target_size)
    imgplot = plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    predicted_class_index = np.argmax(classes)
    if predicted_class_index == 0:
        return "Paper"
    elif predicted_class_index == 1:
        return "Rock"
    elif predicted_class_index == 2:
        return "Scissors"
