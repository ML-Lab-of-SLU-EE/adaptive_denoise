"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import numpy as np
import os
import cv2
import tensorflow as tf
import random
from glob import glob
from tensorflow import keras


class Image_data:

    def __init__(self, img_size, channels, dataset_path, domain_list, augment_flag):
        self.img_height = img_size
        self.img_width = img_size
        self.channels = channels
        self.augment_flag = augment_flag

        self.dataset_path = dataset_path
        self.domain_list = domain_list

        self.images = []
        self.shuffle_images = []
        self.domains = []


    def image_processing(self, filename, filename2, domain):

        x = tf.io.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize(x_decode, [self.img_height, self.img_width])
        img = preprocess_fit_train_image(img)

        x = tf.io.read_file(filename2)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img2 = tf.image.resize(x_decode, [self.img_height, self.img_width])
        img2 = preprocess_fit_train_image(img2)

        if self.augment_flag :
            seed = random.randint(0, 2 ** 31 - 1)
            condition = tf.greater_equal(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), 0.5)

            augment_height_size = self.img_height + (30 if self.img_height == 256 else int(self.img_height * 0.1))
            augment_width_size = self.img_width + (30 if self.img_width == 256 else int(self.img_width * 0.1))

            img = tf.cond(pred=condition,
                          true_fn=lambda : augmentation(img, augment_height_size, augment_width_size, seed),
                          false_fn=lambda : img)

            img2 = tf.cond(pred=condition,
                          true_fn=lambda: augmentation(img2, augment_height_size, augment_width_size, seed),
                          false_fn=lambda: img2)

        return img, img2, domain

    def preprocess(self):
        # self.domain_list = ['tiger', 'cat', 'dog', 'lion']

        for idx, domain in enumerate(self.domain_list):
            image_list = glob(os.path.join(self.dataset_path, domain) + '/*.png') + glob(os.path.join(self.dataset_path, domain) + '/*.jpg')
            shuffle_list = random.sample(image_list, len(image_list))
            domain_list = [[idx]] * len(image_list)  # [ [0], [0], ... , [0] ]

            self.images.extend(image_list)
            self.shuffle_images.extend(shuffle_list)
            self.domains.extend(domain_list)


class new_Image_data:

    def __init__(self, img_size, channels, dataset_path, domain_list, augment_flag):
        self.img_height = img_size
        self.img_width = img_size
        self.channels = channels
        self.augment_flag = augment_flag

        self.dataset_path = dataset_path
        self.domain_list = domain_list

    def image_processing_multi(self, noise, fake_haze_gt, haze_domain, fake_rain_gt, rain_domain, fake_dark_gt, dark_domain, clean, clean_domain):

        x = tf.io.read_file(noise)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize(x_decode, [self.img_height, self.img_width])
        noise_img = preprocess_fit_train_image(img)

        x = tf.io.read_file(fake_haze_gt)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img2 = tf.image.resize(x_decode, [self.img_height, self.img_width])
        fake_haze_gt_img = preprocess_fit_train_image(img2)

        x = tf.io.read_file(fake_rain_gt)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img3 = tf.image.resize(x_decode, [self.img_height, self.img_width])
        fake_rain_gt_img = preprocess_fit_train_image(img3)

        x = tf.io.read_file(fake_dark_gt)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img4 = tf.image.resize(x_decode, [self.img_height, self.img_width])
        fake_dark_gt_img = preprocess_fit_train_image(img4)

        x = tf.io.read_file(clean)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img5 = tf.image.resize(x_decode, [self.img_height, self.img_width])
        clean_img = preprocess_fit_train_image(img5)

        if self.augment_flag :
            seed = random.randint(0, 2 ** 31 - 1)
            condition = tf.greater_equal(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), 0.5)

            augment_height_size = self.img_height + (30 if self.img_height == 256 else int(self.img_height * 0.1))
            augment_width_size = self.img_width + (30 if self.img_width == 256 else int(self.img_width * 0.1))

            noise_img = tf.cond(pred=condition,
                          true_fn=lambda: augmentation(img, augment_height_size, augment_width_size, seed),
                          false_fn=lambda: noise_img)

            fake_haze_gt_img = tf.cond(pred=condition,
                          true_fn=lambda: augmentation(img2, augment_height_size, augment_width_size, seed),
                          false_fn=lambda: fake_haze_gt_img)

            fake_rain_gt_img = tf.cond(pred=condition,
                           true_fn=lambda: augmentation(img2, augment_height_size, augment_width_size, seed),
                           false_fn=lambda: fake_rain_gt_img)

            fake_dark_gt_img = tf.cond(pred=condition,
                           true_fn=lambda: augmentation(img2, augment_height_size, augment_width_size, seed),
                           false_fn=lambda: fake_dark_gt_img)

            clean_img = tf.cond(pred=condition,
                           true_fn=lambda: augmentation(img2, augment_height_size, augment_width_size, seed),
                           false_fn=lambda: clean_img)


        return noise_img, fake_haze_gt_img, haze_domain, fake_rain_gt_img, rain_domain, fake_dark_gt_img, dark_domain, clean_img, clean_domain

    def preprocess_multi(self):
        # self.domain_list = ['tiger', 'cat', 'dog', 'lion']
        # self.domain_list = ['noise', 'fake_haze', 'fake_rain', 'fake_dark', 'clean']
        self.noise = sorted(glob(os.path.join(self.dataset_path, 'noise') + '/*.png') + glob(
            os.path.join(self.dataset_path, 'noise') + '/*.jpg') + glob(
            os.path.join(self.dataset_path, 'noise') + '/*.jpeg'))

        self.fake_haze = sorted(glob(os.path.join(self.dataset_path, 'fake_haze') + '/*.png') + glob(
            os.path.join(self.dataset_path, 'fake_haze') + '/*.jpg') + glob(
            os.path.join(self.dataset_path, 'fake_haze') + '/*.jpeg'))
        self.haze_domain = [[0]] * len(self.fake_haze)

        self.fake_rain = sorted(glob(os.path.join(self.dataset_path, 'fake_rain') + '/*.png') + glob(
            os.path.join(self.dataset_path, 'fake_rain') + '/*.jpg') + glob(
            os.path.join(self.dataset_path, 'fake_rain') + '/*.jpeg'))
        self.rain_domain = [[1]] * len(self.fake_rain)

        self.fake_dark = sorted(glob(os.path.join(self.dataset_path, 'fake_dark') + '/*.png') + glob(
            os.path.join(self.dataset_path, 'fake_dark') + '/*.jpg') + glob(
            os.path.join(self.dataset_path, 'fake_dark') + '/*.jpeg'))
        self.dark_domain = [[2]] * len(self.fake_dark)

        self.clean = sorted(glob(os.path.join(self.dataset_path, 'clean') + '/*.png') + glob(
            os.path.join(self.dataset_path, 'clean') + '/*.jpg') + glob(
            os.path.join(self.dataset_path, 'clean') + '/*.jpeg'))
        self.clean_domain = [[3]] * len(self.clean)


def adjust_dynamic_range(images, range_in, range_out, out_dtype):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images = tf.clip_by_value(images, range_out[0], range_out[1])
    images = tf.cast(images, dtype=out_dtype)
    return images

def preprocess_fit_train_image(images):
    images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
    return images

def postprocess_images(images):
    images = adjust_dynamic_range(images, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
    images = tf.cast(images, dtype=tf.dtypes.uint8)
    return images

def load_images(image_path, img_size, img_channel):
    x = tf.io.read_file(image_path)
    x_decode = tf.image.decode_jpeg(x, channels=img_channel, dct_method='INTEGER_ACCURATE')
    img = tf.image.resize(x_decode, [img_size, img_size])
    img = preprocess_fit_train_image(img)

    return img

def augmentation(image, augment_height, augment_width, seed):
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize(image, [augment_height, augment_width])
    image = tf.image.random_crop(image, ori_image_shape, seed=seed)
    return image

def load_test_image(image_path, img_width, img_height, img_channel):

    if img_channel == 1 :
        img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    else :
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, dsize=(img_width, img_height))

    if img_channel == 1 :
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
    else :
        img = np.expand_dims(img, axis=0)

    img = img/127.5 - 1

    return img

def save_images(images, size, image_path):
    # size = [height, width]
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return ((images+1.) / 2) * 255.0


def imsave(images, size, path):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def return_images(images, size) :
    x = merge(images, size)

    return x

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')

def pytorch_xavier_weight_factor(gain=0.02, uniform=False) :

    factor = gain * gain
    mode = 'fan_avg'

    return factor, mode, uniform

def pytorch_kaiming_weight_factor(a=0.0, activation_function='relu') :

    if activation_function == 'relu' :
        gain = np.sqrt(2.0)
    elif activation_function == 'leaky_relu' :
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif activation_function =='tanh' :
        gain = 5.0 / 3
    else :
        gain = 1.0

    factor = gain * gain
    mode = 'fan_in'

    return factor, mode

def automatic_gpu_usage() :
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def multiple_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096),
                 tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate

def read_cyclegan_ckpt_file(generator, generator_ema, discriminator, checkpoint_dir, name='CycleGAN', phase='train'):
    if phase == 'train':
        ckpt = tf.train.Checkpoint(GA2B=generator, GA2B_ema=generator_ema, D_A=discriminator)
    if phase == 'test':
        ckpt = tf.train.Checkpoint(GA2B_ema=generator_ema)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print('Latest %s checkpoint restored !' % name)
    else:
        print('Not restoring %s model from checkpoint !' % name)

