# coding: utf-8

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


weights_path = 'small_model.h5'
model = load_model(weights_path, custom_objects={'PSNRLoss': PSNRLoss})
model.summary()


def predict(data_x):
    return model.predict(data_x, batch_size=8, verbose=1)


def preprocess_data(path):
    data = transform(path)
    width, height = data.shape[1:-1]
    scale = 4
    a, b = find_div(width, scale), find_div(height, scale)
    data = data[:, :a, :b]
    data = down_sample(data, 2)
    data = up_sample(data, 2)
    return data

root = 'images/ground_truth/'
result = 'images/results/'
scaled = 'images/scaled/'

result = []
for path in os.listdir(root):
    data_x = preprocess_data(root+path)
    start_time = time.time()
    array = predict(data_x)
    run_time = time.time() - start_time
    result.append(run_time)
    print(array.shape)
    transform(array).save(result+path, quality=90)
    transform(data_x).save(scaled+path, quality=90)

print('Average runtime:', sum(result)/len(result))