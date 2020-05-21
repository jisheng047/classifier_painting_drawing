# built-in libraries
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
import IPython.display as display
from PIL import Image

# config
import yaml
import pathlib

warnings.filterwarnings('ignore')

AUTOTUNE = tf.data.experimental.AUTOTUNE
CLASS_NAMES = np.array(["painting", "photograph"])
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 1000
EVALUATION_INTERVAL = 200

def loadConfiguration():
    with open("config.yml", "r") as yaml_config:
        config = yaml.load(yaml_config, Loader=yaml.FullLoader)
    return config
   
def loadData(data_path):
    return pathlib.Path(data_path)

def convertSample(file_path):
    label = getLabel(file_path)
    img = tf.io.read_file(file_path)
    img = decodeImage(img)
    return img, label

def getLabel(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decodeImage(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def prepareForTraining(pre_dataset, cache=False, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            pre_dataset = pre_dataset.cache(cache)
        else:
            pre_dataset = pre_dataset.cache()

    # global SHUFFLE_BUFFER_SIZE
    # shuffle_buffer_size = SHUFFLE_BUFFER_SIZE

    dataset = pre_dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    # dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    # dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

def showBatch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(9):
        ax = plt.subplot(3,3,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
    plt.show()

def processSamples(data_dir):
    list_ds_train = tf.data.Dataset.list_files(str(data_dir/'train/*/*'))
    list_ds_valid = tf.data.Dataset.list_files(str(data_dir/'valid/*/*'))
    list_ds_test = tf.data.Dataset.list_files(str(data_dir/'test/*/*'))

    labeled_ds_train = list_ds_train.map(convertSample, num_parallel_calls=AUTOTUNE)
    labeled_ds_valid = list_ds_valid.map(convertSample, num_parallel_calls=AUTOTUNE)
    labeled_ds_test = list_ds_test.map(convertSample, num_parallel_calls=AUTOTUNE)

    train_batches = prepareForTraining(labeled_ds_train)
    validation_batches = prepareForTraining(labeled_ds_valid)
    test_batches = prepareForTraining(labeled_ds_test)

    return train_batches, validation_batches, test_batches
    
def showLayers(model):
    # VGG16 models architecture
    # index 0: input layer
    # index 1 - 3: block 1
    # index 4 - 6: block 2
    # index 7 - 10: block 3
    # index 11 - 14: block 4
    # index 15 - 18: block 5
    for index, layer in enumerate(model.layers):
        print("Index: {0}, Layer: {1}".format(index,layer))
        # layer.trainable = False


"""
Transfer learning model VGG16 for specific task.
After 92% accuracy, freezing block 1-3 and fine-tuning block 4-5-6.
"""

def main():
    config = loadConfiguration()
    data_dir = loadData(data_path = config["path"]["dataset"])
    train_batches, validation_batches, test_batches = processSamples(data_dir)
    
    # Setting GPU for Keras
    # config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
    # sess = tf.Session(config=config)
    # keras.backend.set_session(sess)

    # image_batch, label_batch = next(iter(train_batches))
    # print(image_batch)

    # Take 1 batch: batch_size images
    for image_batch, label_batch in train_batches.take(1):
        # print(image_batch)
        pass

    ### Image batch
    print("============================= Load Image Batch       =============================")
    print("Image batch: {0}".format(image_batch.shape))
    ### Create Base Model from pre-trained VGG16 convnet
    base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")


    print("============================= Feature batch phrase   =============================")
    feature_batch = base_model(image_batch)
    print("Feature batch: {0}".format(feature_batch.shape))

    

    ### Feature batch average
    print("============================= Global average phrase  =============================")
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print("Feature batch average: {0}".format(feature_batch_average.shape))

    
    ### Prediction batch shape
    print("============================= Prediction phrase      =============================")
    prediction_layer = tf.keras.layers.Dense(units= 2, activation= "softmax")
    prediction_batch = prediction_layer(feature_batch_average)
    print("Prediction batch shape: {0}".format(prediction_batch.shape))
    print("Show 1 value: {0}".format(prediction_batch[0][0]))

    print("============================= Freezing VGG16         =============================")
    base_model.trainable = False
    # print(base_model.summary())

    print("============================= Stacking Sequential    =============================")
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])


    print("============================= Compiling Model        =============================")
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    print("============================= Summary                 =============================")
    print(model.summary())

    # print("============================= Show Layers             =============================")
    # print(showLayers(base_model))


    print("============================= Train model               =============================")
    initial_epochs = 10
    # Validation steps based on batch size
    validation_steps = 625
    test_steps = 625

    # history = model.fit(train_batches, epochs = initial_epochs)
    history = model.fit(train_batches, epochs = initial_epochs, validation_data = validation_batches, validation_steps=validation_steps)
    loss, accuracy = model.evaluate(test_batches, steps = test_steps)

    print("============================= Test Result               =============================")
    print("\nTest loss: {0}".format(loss))
    print("\nTest accuracy: {0}".format(accuracy))


    # print("History: {}".format(history.history))
    # image_batch, label_batch = next(iter(train_batches))
    # showBatch(image_batch.numpy(), label_batch.numpy())
    
    model.save("./model/model_type01_01", save_format="h5")
if __name__ == "__main__":
    main()
