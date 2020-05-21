import urllib

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
AUTOTUNE = tf.data.experimental.AUTOTUNE

import tensorflow_docs as tfdocs
import tensorflow_docs.plots

import tensorflow_datasets as tfds

import PIL.Image

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 5)

import numpy as np

def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title('Augmented image')
    plt.imshow(augmented)

    plt.show()

def main():
    image_path = tf.keras.utils.get_file("cat.jpg", "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg")
    PIL.Image.open(image_path)
    image_string=tf.io.read_file(image_path)
    image=tf.image.decode_jpeg(image_string,channels=3)

    # Flipping a image
    # flipped = tf.image.flip_left_right(image)
    # visualize(image, flipped)

    # Grayscale a image
    # grayscaled = tf.image.rgb_to_grayscale(image)
    # visualize(image, tf.squeeze(grayscaled))

    # Saturate a image
    # saturated = tf.image.adjust_saturation(image, 3)
    # visualize(image, saturated)
    
    # Bright a image
    # bright = tf.image.adjust_brightness(image, 0.4)
    # visualize(image, bright)

    # Rotate a image
    # rotated = tf.image.rot90(image)
    # visualize(image, rotated)

    # Center crop the image
    # cropped = tf.image.central_crop(image, central_fraction=0.5)
    # visualize(image,cropped)

    


if __name__ == "__main__":
    main()