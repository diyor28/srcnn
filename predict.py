import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import random

random.seed(1337)
np.random.seed(1337)

import tensorflow as tf

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

from keras.applications.vgg19 import VGG19


class LossModel():
    def __init__(self):
        with tf.device('/cpu:0'):
            self.model = VGG19(include_top=False, input_shape=(None, None, 3))

    def predict(self, array, batch_size):
        with tf.device('/cpu:0'):
            return self.model.predict(array, batch_size=batch_size)
