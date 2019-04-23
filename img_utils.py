import numpy as np
import random
import time
import math
import os

from PIL import Image
from sklearn.utils import shuffle
import keras.backend as K
from keras.applications.vgg19 import preprocess_input
from skimage.color import *
import scipy.ndimage.interpolation as sni

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return 20 * K.log(K.max(y_true)) - 10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def convert_cs(x, input_cs, output_cs, bw=False):
    mean = np.array([123.68, 116.78, 103.94], dtype='float32').reshape((1, 1, 3))
    x = x.copy()
    if input_cs == output_cs:
        return x

    if input_cs == 'lab':
        if bw: x = np.concatenate((x, np.zeros(x.shape[:-1]+(2,))), axis=-1)

        if x.ndim == 4:
            for idx, x_ in enumerate(x):
                x[idx] = lab2rgb(x_ * 128) * 255
        else:
            x = lab2rgb(x * 128) * 255

    elif input_cs == 'rgb':
        if bw:
            x = gray2rgb(rgb2gray(x))
        x *= 255
        x = x.clip(0, 255)

    elif input_cs == 'bgr':
        x = x[..., ::-1]
        x += mean
    else:
        print('Invalid color space:', input_cs)

    if output_cs == 'lab':
        if x.ndim == 4:
            for idx, x_ in enumerate(x):
                x[idx] = rgb2lab(x_ / 255) / 128
        else:
            x = rgb2lab(x / 255) / 128
        if bw: x = x[..., :1]

    elif output_cs == 'rgb':
        x = x.clip(0, 255)
        x /= 255
        if bw:
            x = np.expand_dims(rgb2gray(x), axis=-1)
    elif output_cs == 'bgr':
        x = preprocess_input(x)
    else:
        print('Invalid color space:', output_cs)

    return x


def squeeze_image(img, xy):
    img.thumbnail((xy, xy), Image.ANTIALIAS)
    w, h = img.size
    result = Image.new("RGB", (xy, xy))
    result.paste(img, (0, 0, w, h))
    return result


def transform(img, input_space='rgb', output_space='rgb',
              xy=None, keep_dims=False):
    if type(img) == str:
        img = Image.open(img).convert('RGB')

    elif type(img) == np.ndarray:
        x = img[0].copy() if img.ndim == 4 else img.copy()
        x = convert_cs(x, input_space, output_space) * 255
        img = Image.fromarray(x.astype('uint8'))

        if xy and keep_dims:
            img = squeeze_image(img, xy)

        elif xy and not keep_dims:
            img = img.resize((xy, xy))

        elif keep_dims:
            img.thumbnail((xy, xy), Image.ANTIALIAS)

        return img

    if xy and keep_dims:
        img = squeeze_image(img, xy)

    elif xy and not keep_dims:
        img = img.resize((xy, xy))

    elif keep_dims:
        img.thumbnail((xy, xy), Image.ANTIALIAS)

    x = np.asarray(img, dtype='float32') / 255
    x = convert_cs(x, input_cs=input_space, output_cs=output_space)
    x = np.expand_dims(x, axis=0)
    return x


def break_image(x, y, batch_size, window_size):
    total_x, total_y = [], []
    dim = min(x.shape[1], x.shape[2]) - window_size - 1
    if dim <= 0:
        #         print(f'too small image {x.shape}')
        return x, y
    for i in range(batch_size):
        a = random.randint(0, dim)
        b = random.randint(0, dim)
        total_x.append(x[:, a:a + window_size, b:b + window_size, ])
        total_y.append(y[:, a:a + window_size, b:b + window_size])
    return np.vstack(total_x), np.vstack(total_y)


def find_div(n, m):
    q = n // m
    n1 = m * q
    n2 = m * (q - 1)

    if abs(n - n1) < abs(n - n2):
        return n1
    else:
        return n2


def down_sample(array, scale):
    return sni.zoom(array, (1, 1/scale, 1/scale, 1), order=2)


def up_sample(array, scale):
    return sni.zoom(array, (1, scale, scale, 1), order=2)


def crop(img, slices=2, use_array=True, return_array=True):
    if use_array: img = transform(img)
    images = []
    imgwidth, imgheight = img.size
    height, width = int(imgheight / slices), int(imgwidth / slices)
    for i in range(slices):
        for j in range(slices):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            images.append(img.crop(box))
    if return_array: images = np.vstack([transform(i) for i in images])
    return images


def concat_images(imgs, use_array=True, return_array=True):
    if use_array: imgs = [transform(i) for i in imgs]
    w, h = imgs[0].size
    slices = int(math.sqrt(len(imgs)))
    n = 0
    result = Image.new("RGB", (w * slices, h * slices))
    for i in range(slices):
        for j in range(slices):
            img = imgs[n]
            n += 1
            box = (j * w, i * h)
            result.paste(img, box)
    if return_array: result = transform(result)
    return result


def generator(label_path, network, batch_size=32, xy=None):
    labels = [label_path + i for i in os.listdir(label_path)]
    steps = len(labels) // batch_size
    print(steps)
    while True:
        for i in range(steps):
            data_x, data_y = [], []
            labels = shuffle(labels)
            for k in range(batch_size):
                y = transform(labels[k], xy=xy, output_space='bgr')
                x = down_sample(y, 4)
                x = up_sample(x, 4)
                data_x.append(x), data_y.append(y)

            data_y = np.vstack(data_y)
            loss = network.predict(data_y, batch_size=batch_size)
            yield np.vstack(data_x), loss


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes
