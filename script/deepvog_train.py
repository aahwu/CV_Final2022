import os
import numpy as np
from DeepVOG_model import DeepVOG_net
import cv2
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset_path = r'C:\Users\ShaneWu\OneDrive\Desktop\Hw(senior)\CV\Final\CV22S_Ganzin\dataset\public\S1'

IMAGE_SIZE_H = 240
IMAGE_SIZE_W = 320
BATCH_SIZE = 1
NUM_CLASSES = 20
# DATA_DIR = "./instance-level_human_parsing/instance-level_human_parsing/Training"
NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 50

train_images = glob(os.path.join(dataset_path, "01/0.jpg"))
train_masks = glob(os.path.join(dataset_path, "01/0.png"))
# train_images = sorted(train_images)
# train_masks = sorted(train_masks)
# val_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[
#     NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
# ]
# val_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[
#     NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
# ]


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE_H, IMAGE_SIZE_W])
    else:
        image = tf.image.decode_jpeg(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE_H, IMAGE_SIZE_W])
        image = image / 255
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


train_dataset = data_generator(train_images, train_masks)
# val_dataset = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)

# print("Val Dataset:", val_dataset)

model = DeepVOG_net(input_shape = (240, 320, 3), filter_size= (10,10))
model.summary()
loss = keras.losses.BinaryCrossentropy()
model.compile(
    optimizer=keras.optimizers.SGD(),
    loss=loss,
    metrics=["accuracy"],
)
model.fit(train_dataset, epochs=5)

# test_path = r'C:\Users\ShaneWu\OneDrive\Desktop\Hw(senior)\CV\Final\CV22S_Ganzin\model'
# model.save(os.path.join(test_path, "test"))
# # model = keras.models.load_model(os.path.join(test_path, "test"))
# image = read_image(os.path.join(test_path, "0.jpg"))
# # # inference
# # image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_LINEAR)
# img = np.zeros((1, 240, 320, 3))
# print(image.shape)
# print(image)
# # img[:,:,:,:] = (image[:, :, 0]).reshape(1, 240, 320, 1)
# # print(img.shape)
# prediction = model.predict(np.expand_dims((image), axis=0))
# print(prediction.shape)
# print(prediction)
# prediction_thres = prediction > 0.5
# # --output
# prediction_label = prediction_thres.astype(np.uint8) * 255
# prediction_label = cv2.resize(prediction_label[0,:,:,:], (640, 480), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite(os.path.join(test_path, "test.png"), prediction_label[:,:,1])