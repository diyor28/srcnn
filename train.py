# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import matplotlib.pyplot as plt

import time

from PIL import Image

import argparse

parser = argparse.ArgumentParser(description='Configure seed')

parser.add_argument('-steps', '--steps', type=int, default=1000, help='Number of episodes')
parser.add_argument('-batch', '--batch', type=int, default=4, help='Batch size')
parser.add_argument('-lr', '--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('-seed', '--seed', type=int, default=1337, help='Random seed')

args = parser.parse_args()

import numpy as np
import random
random.seed(1337)
np.random.seed(1337)

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

init_op = tf.global_variables_initializer()
sess = tf.Session(config=config)
sess.run(init_op)


from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.losses import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau

from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.optimizers import Adam

from keras.utils.generic_utils import get_custom_objects
import math

from predict import LossModel
from img_utils import *


class ModelCheckPoint(Callback):
    def __init__(self, model, path):
        self.network = model
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.network.save(self.path)

class FlatTanh(Activation):
    
    def __init__(self, activation, **kwargs):
        super(FlatTanh, self).__init__(activation, **kwargs)
        self.__name__ = 'flat_tanh'

def flat_tanh(x):
    return K.maximum(K.minimum(4., x), -4.)


get_custom_objects().update({'flat_tanh': FlatTanh(flat_tanh)})


def create_model():
    input_tensor = Input(shape=(None, None, 3))

    conv = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    conv_input = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv)
    conv = Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(conv_input)
    conv = BatchNormalization()(conv)

    conv_0 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu')(conv)
    conv_0 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu')(conv_0)
    conv_0 = Conv2D(96, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(conv_0)
    conv_0 = BatchNormalization()(conv_0)

    conv_1 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv_0)
    conv_1 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv_1)
    conv_1 = Conv2D(128, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(conv_1)
    conv_1 = BatchNormalization()(conv_1)

    conv_2 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(conv_1)
    conv_2 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(conv_2)
    conv_2 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(conv_2)
    conv_2 = BatchNormalization()(conv_2)

    conv_2 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(conv_2)
    conv_2 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(conv_2)
    conv_2 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(conv_2)
    conv_2 = BatchNormalization()(conv_2)

    conv_3 = UpSampling2D()(conv_2)
    conv_3 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
    conv_3 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
    conv_3 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
    conv_4 = BatchNormalization()(conv_3)

    conc_0 = Concatenate()([conv_4, conv_0])

    conv_3 = UpSampling2D()(conc_0)
    conv_3 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
    conv_3 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
    conv_3 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
    conv_3 = BatchNormalization()(conv_3)

    conc_1 = Concatenate()([conv_3, conv])

    conv_4 = UpSampling2D()(conc_1)
    conv_4 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv_4)
    conv_4 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv_4)
    conv_4 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv_4)
    conv_4 = BatchNormalization()(conv_4)

    conc_2 = Concatenate()([conv_4, conv_input])

    conv_5 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conc_2)
    conv_5 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv_5)
    conv_5 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv_5)
    conv_5 = BatchNormalization()(conv_5)

    output = Conv2D(3, (3, 3), padding='same', activation=None)(conv_5)

    model = Model(inputs=[input_tensor], outputs=[output])
    model.compile(optimizer=Adam(lr=args.lr), loss='mse', metrics=[PSNRLoss])
    model.summary()
    return model


def perpceptual_loss(model):
    loss_model = VGG19(include_top=False, input_shape=(None, None, 3))
    loss_out = loss_model(model.output)
    full_model = Model(model.input, loss_out)

    loss_model.trainable = False
    for l in loss_model.layers:
        l.trainable = False

    full_model.compile(optimizer=Adam(lr=args.lr), loss='mse', metrics=[PSNRLoss])
    full_model.summary()
    return full_model


network = LossModel()

model = create_model()
weights_path = 'weights/small_model.h5'

try:
    model.load_weights(weights_path)
except Exception as e:
    print(e)

full_model = perpceptual_loss(model)

# x, y, loss = next(gen)

# print(model.evaluate(x=data_x, y=data_y, verbose=1, batch_size=2), 'before training')

gen = generator('D:/dataset/', network=network, batch_size=4, xy=256)
val_gen = generator('images/full_size/ground_truth/', network=network, batch_size=4, xy=256)
x, loss = next(gen)

model_check_point = ModelCheckPoint(model, weights_path)
rl_reduce = ReduceLROnPlateau(monitor='train_loss', factor=0.5, patience=5, verbose=1, mode='min',
                              min_delta=0.0001, cooldown=0, min_lr=0.0000001)
try:
    full_model.fit_generator(gen, steps_per_epoch=1000, epochs=350, callbacks=[model_check_point, rl_reduce],
                             validation_data=val_gen, validation_steps=1)
    # full_model.fit(x=x, y=loss, epochs=100, batch_size=2)
except KeyboardInterrupt:
    print("Interrupted")
    model.save(weights_path)
    print('model successfully saved')

# print(model.evaluate(x=data_x, y=data_y, verbose=1, batch_size=2), 'after training')

# x, y = transform('train_y/461_0.jpg'), transform('train_y/461_0.jpg')
# x, y = break_image(x, y, batch_size=64, window_size=400)
# x = down_sample(x, 4)
# x = up_sample(x, 4)
# loss = network.predict(y, batch_size=4)
#
# data_x, data_y = x, y

# array = model.predict(data_x, batch_size=2, verbose=1)
#
# print(np.max(array[0]))
# print(np.min(array[0]))
#
# predicted = concat_images(array)
# noisy = concat_images(data_x)
# original = concat_images(data_y)
# print(noisy.shape)
# print(predicted.shape)
#
# transform(noisy).save('images/noisy.jpg', quality=95)
# transform(predicted).save('images/predicted.jpg', quality=95)
# transform(original).save('images/ground_truth.jpg', quality=95)
