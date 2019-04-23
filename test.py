# coding: utf-8

import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import time

from PIL import Image
from IPython import display

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

from keras import backend as K

from keras.models import *
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import *
from keras.utils import *
from keras.losses import *

from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam

from keras.utils.generic_utils import get_custom_objects
from predict import LossModel

import math

from img_utils import *


class FlatTanh(Activation):
    
    def __init__(self, activation, **kwargs):
        super(FlatTanh, self).__init__(activation, **kwargs)
        self.__name__ = 'flat_tanh'


def flat_tanh(x):
    return K.maximum(K.minimum(4., x), -4.)


get_custom_objects().update({'flat_tanh': FlatTanh(flat_tanh)})

network = LossModel()
weights_path = 'weights/full_set.h5'
weights_path = 'weights/small_model.h5'
model = load_model(weights_path, custom_objects={'PSNRLoss':PSNRLoss})

model.summary()


def preprocess_data(path):
    data = transform(path)
    width, height = data.shape[1:-1]
    scale = 4
    a, b = find_div(width, scale), find_div(height, scale)
    data = data[:, :a, :b]
    data = down_sample(data, 4)
    data = up_sample(data, 4)
    data = convert_cs(data, input_cs='rgb', output_cs='bgr')
    return data


root = 'images/full_size/ground_truth/'
result = 'images/full_size/results/'
scaled = 'images/full_size/scaled/'

for path in os.listdir(root):
    data_x = preprocess_data(root+path)
    array = model.predict(data_x, batch_size=8, verbose=1)
    print(array.shape)
    transform(array, input_space='bgr', output_space='rgb').save(result+path, quality=90)
    transform(data_x, input_space='bgr', output_space='rgb').save(scaled+path, quality=90)

# print('saving images')
# transform(data_x).save('images/full_size/noisy.jpg', quality=95)
# transform(array).save('images/full_size/predicted.jpg', quality=95)
# transform(data_y).save('images/full_size/ground_truth.jpg', quality=95)
