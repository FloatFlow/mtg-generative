# imports
import itertools
import random
import cv2
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from queue import Queue
from multiprocessing import Pool
import threading
from threading import Lock
import time
from functools import partial

####################################################
## Loss functions
####################################################

def real_discriminator_loss(true, pred):
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - pred))
    return real_loss

def fake_discriminator_loss(true, pred):
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + pred))
    return fake_loss

def generator_loss(true, pred):
    fake_loss = -tf.reduce_mean(pred)
    return fake_loss

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

def image_normalize_read(img_path, img_dim):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_dim, img_dim))
    img = img[:,:,::-1]
    img = img/127.5
    img -= 1
    return img

def img_resize(img, y_dim, x_dim):
    if img.shape[0]*img.shape[1] < y_dim*x_dim:
        img = cv2.resize(img,
                         (y_dim, x_dim),
                         interpolation=CV_INTER_CUBIC)
    else:
        img = cv2.resize(img,
                         (y_dim, x_dim),
                         interpolation=CV_INTER_AREA)
    return img

def crop_square_from_rec(img, img_dim=256):
    # resize based on smallest axis
    smallest_axis = img.shape[:2].index(min(img.shape[:2]))
    scale = img_dim/img.shape[smallest_axis]
    img = img_resize(img,
                     int(img.shape[0]*scale),
                     int(img.shape[1]*scale))

    # crop square from image
    smallest_axis = img.shape[:2].index(min(img.shape[:2]))
    largest_axis = [axis for axis in [0, 1] if axis not smallest_axis][0]
    largest_axis_len = img.shape[largest_axis]
    possible_crops = largest_axis_len-img_dim
    rand_position = np.random.randint(possible_crops)
    if smallest_axis == 0:
        img = img[:, rand_position:rand_position+img_dim, :]
    else:
        img = img[rand_position:rand_position+img_dim, :, :]
    return img

def card_crop(img_path, img_dim=256):
    img = cv2.imread(img_path)
    # resize if too small
    if (img.shape[0] < img_dim) or (img.shape[1] < img_dim):
        img = crop_square_from_rec(img, img_dim)

    # if image is too large
    else:
        random_method = np.random.randint(2)
        # resize so smallest axis = img_dim, then crop
        if random_method == 0:
            img = crop_square_from_rec(img, img_dim)
        else:
            smallest_axis_len = min(img.shape[:2])
            percent = np.random.randint(80, 101)/10
            target_crop_size = int(smallest_axis_len*percent)
            x_positions = img.shape[0]-target_crop_size
            y_positions = img.shape[1]-target_crop_size
            rand_x = np.random.randint(x_positions)
            rand_y = np.random.randint(y_positions)
            img = img[rand_x:rand_x+target_crop_size,
                      rand_y:rand_y+target_crop_size,
                      :]

    # randomly flip horizontally
    flip_val = np.random.randint(2)
    if flip_val == 0:
        img = img[:, ::-1, :]

    # normalize
    img = img[:,:,::-1]
    img = img/127.5
    img = img-1.0
    return img.astype(np.float32)


def label_generator(n_samples, seed):
    sample_list = list(itertools.permutations([1,0,0,0,0])) + list(itertools.permutations([1,1,0,0,0])) + list(itertools.permutations([1,1,1,0,0]))
    random.seed(seed)
    random.shuffle(sample_list)
    return sample_list[:n_samples]

####################################################
## Data Generator
####################################################

class CardGenerator():
    def __init__(self,
                 img_dir,
                 batch_size,
                 n_threads, 
                 n_cpu, 
                 img_dim, 
                 file_type = '.jpg'):
        self.img_dir = img_dir
        self.img_dim = img_dim
        self.file_type = file_type
        self.batch_size = batch_size
        self.file_type = file_type
        self.n_threads = n_threads
        self.n_cpu = n_cpu
        self.generate_data_paths()
        self.queue = Queue(maxsize=4)
        self.lock = Lock()
        self.threads = []
        self.run = True
        self.pools = [Pool(1) for _ in range(self.n_threads)]
        self.counter = 0
        self.queue_len = 0

        for i in range(self.n_threads):
            self.threads.append(threading.Thread(target=self.multicpu_read,
                                                 args=(self.pools[i], )))
            self.threads[i].daemon = True
            self.threads[i].start()   

    def generate_data_paths(self):
        img_paths = []
        img_labels = []
        for file in os.listdir(self.img_dir):
            if file.endswith(self.file_type):
                img_paths.append(os.path.join(self.img_dir, file))
                label = os.path.basename(file).split('.')[0].split('_')[-1]
                img_labels.append(label)
        self.n_samples = len(img_paths)

        # convert labels to one-hot encoded
        lookup_dict = {'W':[1,0,0,0,0], 'B':[0,1,0,0,0], 'U':[0,0,1,0,0], 'G':[0,0,0,1,0], 'R':[0,0,0,0,1]}
        one_hot_labels = []
        for str_label in img_labels:
            temp = [0,0,0,0,0]
            for k, v in lookup_dict.items():
                if k in str_label:
                    temp = [sum(x) for x in zip(temp, v)]
            one_hot_labels.append(temp)

        # shuffle and batch
        self.n_batches = self.n_samples//self.batch_size
        self.n_samples = self.n_batches*self.batch_size
        self.path_batches = [img_paths[i:i+self.batch_size] for i in range(0, self.n_samples, self.batch_size)]
        self.label_batches = [one_hot_labels[i:i+self.batch_size] for i in range(0, self.n_samples, self.batch_size)]

    def multicpu_read(self, pool):
        while self.run:

            while self.queue.full():
                time.sleep(0.1)

            if self.queue.full() == False:
                self.lock.acquire()
                idx = self.counter
                self.counter += 1
                if self.counter >= self.n_batches:
                    self.counter = 0
                self.queue_len += 1
                self.lock.release()

                numpy_batch = list(pool.imap(partial(card_crop, img_dim=self.img_dim),
                                             [img_path for img_path in self.path_batches[idx]]))
                label_batch = self.label_batches[idx]

                self.queue.put((numpy_batch, np.array(label_batch)))

    def next(self):
        while self.run:
            while self.queue.empty():
                time.sleep(0.1)
            self.queue_len -= 1
            return self.queue.get()

    def shuffle(self, seed):
        self.path_batches = [item for sublist in self.path_batches for item in sublist]
        self.label_batches = [item for sublist in self.label_batches for item in sublist]
        concat = list(zip(self.path_batches, self.label_batches))
        random.shuffle(concat)
        img_paths, one_hot_labels = zip(*concat)
        self.path_batches = [img_paths[i:i+self.batch_size] for i in range(0, self.n_samples, self.batch_size)]
        self.label_batches = [one_hot_labels[i:i+self.batch_size] for i in range(0, self.n_samples, self.batch_size)]

    def end(self):
        self.run = False
        for thread in self.threads:
            thread.end()
