# imports
import itertools
import random
import cv2
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from multiprocessing import Process, Queue, Lock, Value
import time
from functools import partial
import keras.backend as K
from keras.utils import Sequence
from PIL import Image
#import colorgram

####################################################
## Loss functions
####################################################

def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def vq_reconstruction_loss(y_true, y_pred, commitment_loss):
    reconstruction_loss = tf.reduce_mean(tf.square(y_pred-y_true)) / reduce_var(y_true)
    return reconstruction_loss + commitment_loss

def hinge_real_discriminator_loss(y_true, y_pred):
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - y_pred))
    return real_loss

def hinge_fake_discriminator_loss(y_true, y_pred):
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + y_pred))
    return fake_loss

def hinge_generator_loss(y_true, y_pred):
    fake_loss = -tf.reduce_mean(y_pred)
    return fake_loss

def wgan_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def nonsat_fake_discriminator_loss(y_true, y_pred, sample_weight=1):
    return K.mean(K.softplus(y_pred))

def nonsat_real_discriminator_loss(y_true, y_pred, sample_weight=1):
    return K.mean(K.softplus(-y_pred))

def nonsat_generator_loss(y_true, y_pred, sample_weight=1):
    return K.mean(K.softplus(-y_pred))

#r1/r2 gradient penalty
def gradient_penalty_loss(y_true, y_pred, averaged_samples, sample_weight=1):
    y_true = None
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_penalty = K.sqrt(gradient_penalty + K.epsilon())
    # weight * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty * sample_weight)

def vq_latent_loss(y_true, y_pred, beta=1, sample_weight=1):
    y_true = None
    latent_dim = K.int_shape(y_pred)[-1]//2
    z_e = y_pred[..., :latent_dim]
    z_q = y_pred[..., latent_dim:]
    vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - z_q)**2)
    commit_loss = tf.reduce_mean((z_e - tf.stop_gradient(z_q))**2)
    latent_loss = tf.identity(vq_loss + beta * commit_loss, name="latent_loss")
    return latent_loss

def zq_norm(y_true, y_pred):
    del y_true
    latent_dim = K.int_shape(y_pred)[-1]//2
    z_q = y_pred[..., latent_dim:]
    return tf.reduce_mean(tf.norm(z_q, axis=-1))

def ze_norm(y_true, y_pred):
    del y_true
    latent_dim = K.int_shape(y_pred)[-1]//2
    z_e = y_pred[..., :latent_dim]
    return tf.reduce_mean(tf.norm(z_e, axis=-1))

def pixelcnn_accuracy(y_true, y_pred):
    """Train the PixelCNN and monitor prediction accuracy"""
    size = K.int_shape(y_pred)[-2]
    k = K.int_shape(y_pred)[-1]
    y_true = K.reshape(y_true, (-1, size * size))
    y_pred = K.reshape(y_pred, (-1, size * size, k))
    acc = K.cast(
        K.equal(
            y_true,
            K.cast(K.argmax(y_pred, axis=-1), "float32")
            ),
        "float32"
        )
    return acc

def kl_loss(y_true, y_pred, z_mean, z_log_var):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss

####################################################
## Image Preprocessing and loading
####################################################

def card_batch_collector(dir, batch_size, seed, file_type='.jpg'):
    # collect all image paths and labels
    img_paths = []
    img_labels = []
    for file in os.listdir(dir):
        if file.endswith(file_type):
            img_paths.append(os.path.join(dir, file))
            label = os.path.basename(file).split('.')[0].split('_')[-1]
            img_labels.append(label)
    df = pd.DataFrame({'path':img_paths, 'label':img_labels})

    # convert labels to one-hot encoded
    lookup_dict = {'W':[1,0,0,0,0], 'B':[0,1,0,0,0], 'U':[0,0,1,0,0], 'G':[0,0,0,1,0], 'R':[0,0,0,0,1]}
    one_hot_labels = []
    for str_label in df['label']:
        temp = [0,0,0,0,0]
        for k, v in lookup_dict.items():
            if k in str_label:
                temp = [sum(x) for x in zip(temp, v)]
        one_hot_labels.append(temp)
    df['label'] = one_hot_labels

    # shuffle and batch
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_batches = df.shape[0]//batch_size
    n_samples = n_batches*batch_size
    path_batches = [df['path'].values[i:i+batch_size] for i in range(0, n_samples, batch_size)]
    label_batches = [df['label'].values[i:i+batch_size] for i in range(0, n_samples, batch_size)]
    zipped_batches = [(p, l) for p, l in zip(path_batches, label_batches)]

    return zipped_batches

def img_resize(img, y_dim, x_dim):
    img = Image.fromarray(img)
    img = img.resize((y_dim, x_dim), Image.LANCZOS)
    return np.array(img)

def batch_resize(batch, dim):
    return np.stack([img_resize(img, dim, dim) for img in batch])

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.UMat(np.array(image, dtype=np.uint8))
    image = cv2.LUT(image, table).get()
    return image

def convert_temp(image, temp):
    image = Image.fromarray(image)
    kelvin_table = {
    1000: (255,56,0),
    1500: (255,109,0),
    2000: (255,137,18),
    2500: (255,161,72),
    3000: (255,180,107),
    3500: (255,196,137),
    4000: (255,209,163),
    4500: (255,219,186),
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)}
    r, g, b = kelvin_table[temp]
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )
    image = np.array(image.convert('RGB', matrix))
    return image

def crop_square_from_rec(img, img_dim=512):
    assert 0 not in img.shape
    smallest_axis = np.argmin(img.shape[:2])
    largest_axis = np.argmax(img.shape[:2])
    smallest_shape = img.shape[smallest_axis]
    largest_shape = img.shape[largest_axis]
    n_positions = largest_shape - smallest_shape
    if n_positions > 0:
        jiggle = np.random.randint(n_positions)
        if smallest_axis == 0:
            img = img[:, jiggle:jiggle+smallest_shape, :]
        else:
            img = img[jiggle:jiggle+smallest_shape, :, :]
    img = img_resize(img, img_dim, img_dim)
    return img

def card_crop(img_path, img_dim=256):
    #img = cv2.imread(img_path)[:, :, ::-1]
    img = Image.open(img_path)
    img = np.array(img.convert('RGB'))
    # randomly decide gamma and warmth augmentations
    change_warmth = np.random.randint(2)
    if change_warmth == 0:
        warmth = random.choice([5000, 5500, 6000, 6500, 7000, 7500])
        img = convert_temp(img, warmth)
    change_gamma = np.random.randint(2)
    if change_gamma == 0:
        gamma = random.uniform(0.7, 1.3)
        img = adjust_gamma(img, gamma)
    ## image cropping and resizing
    img = crop_square_from_rec(img, img_dim=img_dim)

    # randomly flip horizontally
    flip_val = np.random.randint(2)
    if flip_val == 0:
        img = img[:, ::-1, :]

    # normalize
    return img

def label_generator(n_samples, n_shared_classes=1, single_class_weight=4):
    class_labels = np.tile(np.identity(5), (single_class_weight*5, 1))

    if (n_shared_classes > 1):
       for i in range(1, 5):
           new_double_class = np.identity(5) + np.roll(np.identity(5), i, axis=-1)
           class_labels = np.concatenate(
               [class_labels, new_double_class],
               axis=0
               )
    np.random.shuffle(class_labels)
    return class_labels[:n_samples, :]

def onehot_label_encoder(str_label):
    # convert labels to one-hot encoded
    lookup_dict = {'W':np.array([1,0,0,0,0]),
                   'B':np.array([0,1,0,0,0]),
                   'U':np.array([0,0,1,0,0]),
                   'G':np.array([0,0,0,1,0]),
                   'R':np.array([0,0,0,0,1])}

    one_hot_label = np.zeros(shape=(5, ))
    for k, v in lookup_dict.items():
        if k in str_label:
            one_hot_label += v
    if np.sum(one_hot_label) == 0:
        one_hot_label = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    return one_hot_label/np.sum(one_hot_label)

def onehot_label_decoder(one_hot_label):
    # convert labels to one-hot encoded
    str_label = []
    for i, label in enumerate(one_hot_label):
        if (i == 0) and (label == 1):
            str_label.append('W')
        if (i == 1) and (label == 1):
            str_label.append('B')
        if (i == 2) and (label == 1):
            str_label.append('U')
        if (i == 3) and (label == 1):
            str_label.append('G')
        if (i == 4) and (label == 1):
            str_label.append('R')
    return str(str_label)

def downsample_batch(img_batch, scale):
    original_size = img_batch.shape[1]
    return np.stack(
        [np.array(Image.fromarray(img).resize((original_size//scale, original_size//scale), Image.LANCZOS)) for img in img_batch]
        )

####################################################
## Data Generator
####################################################

class ImgGenerator():
    def __init__(
        self,
        img_dir,
        batch_size,
        n_cpu,
        img_dim,
        labels=True,
        multiscale=False,
        file_type=('.jpg', '.png', '.jpeg', '.mp4')
        ):
        self.img_dir = img_dir
        self.img_dim = img_dim
        self.file_type = file_type
        self.batch_size = batch_size
        self.multiscale = multiscale
        self.n_cpu = n_cpu
        self.queue = Queue(maxsize=self.n_cpu*4)
        self.lock = Lock()
        self.run = True
        self.counter = Value('i', 0)
        self.generate_data_paths()

        for _ in range(self.n_cpu):
             p = Process(target=self.fetch_batch)
             p.start()

    def generate_data_paths(self):
        img_paths = []
        for file in os.listdir(self.img_dir):
            if file.endswith(self.file_type):
                img_paths.append(os.path.join(self.img_dir, file))

        # store paths and labels
        self.df = pd.DataFrame({'paths':img_paths})
        self.n_batches = (self.df.shape[0]//self.batch_size) - 1
        self.indices = np.arange(self.n_batches)

    def fetch_batch(self):
        while self.run:

            while self.queue.full():
                time.sleep(0.1)

            self.lock.acquire()
            idx = self.counter.value
            self.counter.value += 1
            if self.counter.value >= self.n_batches:
                self.counter.value = 0
                self.shuffle()
            self.lock.release()

            try:
                positions = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
                numpy_batch = np.array([card_crop(img_path, self.img_dim) for img_path in self.df['paths'].iloc[positions]])

                if numpy_batch.shape == (self.batch_size, self.img_dim, self.img_dim, 3):
                    #fake_labels = label_generator(self.batch_size)
                    fake_labels = np.full(shape=(self.batch_size, 5), fill_value=0.2)
                    if self.multiscale:
                        multibatch = [(numpy_batch/127.5)-1]
                        for i in [2, 4, 8, 16, 32, 64]:
                            multibatch.append((downsample_batch(numpy_batch, i)/127.5)-1)
                        self.queue.put((multibatch, fake_labels))
                    else:
                        self.queue.put(((numpy_batch/127.5)-1, fake_labels))
            except ValueError:
                #print("Warning: Batch Dropped")
                continue

    def next(self):
        while self.queue.empty():
            time.sleep(0.1)
        return self.queue.get()

    def shuffle(self):
        random.shuffle(self.indices)

    def end(self):
        self.run = False
        self.queue.close()

class CardGenerator():
    def __init__(self,
                 img_dir,
                 batch_size,
                 n_cpu,
                 img_dim,
                 multiscale=False,
                 file_type=('.jpg', '.png', '.jpeg')):
        self.img_dir = img_dir
        self.img_dim = img_dim
        self.file_type = file_type
        self.batch_size = batch_size
        self.n_cpu = n_cpu
        self.multiscale = multiscale
        self.queue = Queue(maxsize=self.n_cpu*4)
        self.lock = Lock()
        self.run = True
        self.counter = Value('i', 0)
        self.generate_data_paths()

        for _ in range(self.n_cpu):
             p = Process(target=self.fetch_batch)
             p.start()

    def generate_data_paths(self):
        img_paths = []
        img_labels = []
        for file in os.listdir(self.img_dir):
            if file.endswith(self.file_type):
                img_paths.append(os.path.join(self.img_dir, file))
                label = os.path.basename(file).split('.')[0].split('_')[-1]
                img_labels.append(label)

        # store paths and labels
        self.df = pd.DataFrame({'paths':img_paths, 'labels':img_labels})
        self.n_batches = (self.df.shape[0]//self.batch_size) - 1
        self.indices = np.arange(self.n_batches)

    def fetch_batch(self):
        while self.run:

            while self.queue.full():
                time.sleep(0.1)

            self.lock.acquire()
            idx = self.counter.value
            self.counter.value += 1
            if self.counter.value >= self.n_batches:
                self.counter.value = 0
                self.shuffle()
            self.lock.release()

            try:
                positions = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
                numpy_batch = np.array([card_crop(img_path, self.img_dim) for img_path in self.df['paths'].iloc[positions]])
                label_batch = np.array([onehot_label_encoder(label) for label in self.df['labels'].iloc[positions]])

                if numpy_batch.shape == (self.batch_size, self.img_dim, self.img_dim, 3):
                    fake_labels = label_generator(self.batch_size)
                    if self.multiscale:
                        multibatch = [(numpy_batch/127.5)-1]
                        for i in [2, 4, 8, 16, 32, 64]:
                            multibatch.append((downsample_batch(numpy_batch, i)/127.5)-1)
                        self.queue.put((multibatch, fake_labels))
                    else:
                        self.queue.put(((numpy_batch/127.5)-1, fake_labels))
            except ValueError:
                #print("Warning: Batch Dropped")
                continue

    def next(self):
        while self.queue.empty():
            time.sleep(0.1)
        return self.queue.get()

    def shuffle(self):
        self.df = self.df.sample(frac=1.0).reset_index(drop=True)

    def end(self):
        self.run = False
        self.queue.close()

