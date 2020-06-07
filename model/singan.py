import cv2
from PIL import Image
import os
import numpy as np
from functools import partial
from tqdm import tqdm
import itertools

from keras.layers import Dense, Reshape, Lambda, Multiply, Add, \
    Activation, UpSampling2D, AveragePooling2D, Input, \
    Concatenate, Flatten, Conv2D, Conv2DTranspose
from keras.models import Model
from keras.engine.network import Network
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

from model.utils import ImgGenerator, nonsat_fake_discriminator_loss, nonsat_real_discriminator_loss, nonsat_generator_loss
from model.network_blocks import singan_generator_block, singan_discriminator_block

class SinGAN():
    def __init__(self, 
                 img_dim_x,
                 img_dim_y,
                 img_depth,
                 g_lr,
                 d_lr,
                 batch_size,
                 save_freq,
                 training_dir,
                 validation_dir,
                 checkpoint_dir,
                 testing_dir,
                 z_len,
                 n_classes,
                 n_gpu,
                 n_cpu):
        #override some args
        self.z_len = None
        self.n_classes = None
        self.batch_size = 1
        self.img_dim_x = 512
        self.img_dim_y = 512
        self.img_depth = 3
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.save_freq = save_freq
        self.n_cpu = n_cpu
        self.training_dir = training_dir
        self.validation_dir = validation_dir
        self.checkpoint_dir = checkpoint_dir
        self.testing_dir = testing_dir
        for path in [self.validation_dir,
                     self.checkpoint_dir,
                     self.testing_dir]:
            if not os.path.isdir(path):
                os.makedirs(path)
        self.name = 'singan'
        self.resolutions = [4, 8, 16, 32, 64, 128, 256, 512]
        assert self.resolutions[-1] == self.img_dim_y
        self.alpha = 0.25
        self.build_generator()
        self.build_discriminator()
        self.build_model()
        print("Model Name: {}".format(self.name))

    ###############################
    ## All our architecture
    ###############################

    def build_generator(self):
        noise_inputs = [Input((dim, dim, self.img_depth)) for dim in self.resolutions]
        ch = 16
        fake_img_4 = singan_generator_block(noise_inputs[0], noise_inputs[0], ch)
        x = UpSampling2D(2, interpolation='bilinear')(fake_img_4)
        fake_img_8 = singan_generator_block(x, noise_inputs[1], ch)
        x = UpSampling2D(2, interpolation='bilinear')(fake_img_8)
        fake_img_16 = singan_generator_block(x, noise_inputs[2], ch)
        x = UpSampling2D(2, interpolation='bilinear')(fake_img_16)
        fake_img_32 = singan_generator_block(x, noise_inputs[3], ch)
        ch = ch*2
        x = UpSampling2D(2, interpolation='bilinear')(fake_img_32)
        fake_img_64 = singan_generator_block(x, noise_inputs[4], ch)
        x = UpSampling2D(2, interpolation='bilinear')(fake_img_64)
        fake_img_128 = singan_generator_block(x, noise_inputs[5], ch)
        x = UpSampling2D(2, interpolation='bilinear')(fake_img_128)
        fake_img_256 = singan_generator_block(x, noise_inputs[6], ch)
        x = UpSampling2D(2, interpolation='bilinear')(fake_img_256)
        fake_img_512 = singan_generator_block(x, noise_inputs[7], ch)
        self.generator = Model(
            noise_inputs,
            [fake_img_4, fake_img_8, fake_img_16, fake_img_32, fake_img_64, fake_img_128, fake_img_256, fake_img_512]
            )
        print(self.generator.summary())

    def build_discriminator(self):
        img_inputs = [Input((dim, dim, self.img_depth)) for dim in self.resolutions]
        #chs = [64, 64, 64, 64, 32, 32, 32, 32]
        chs = [32, 32, 32, 32, 16, 16, 16, 16]
        assert len(chs) == len(self.resolutions)

        labels = [singan_discriminator_block(inputs, ch) for inputs, ch in zip(img_inputs, chs)]
        self.discriminator = Model(img_inputs, labels)
        self.frozen_discriminator = Network(img_inputs, labels)
        print(self.discriminator.summary())

    def build_model(self):
        # build discriminator model
        real_imgs = [Input((dim, dim, self.img_depth)) for dim in self.resolutions]
        fake_imgs = [Input((dim, dim, self.img_depth)) for dim in self.resolutions]
        real_labels = self.discriminator(real_imgs)
        fake_labels = self.discriminator(fake_imgs)
        imgs = real_imgs + fake_imgs
        labels = real_labels + fake_labels
        self.discriminator_model = Model(imgs, labels)
        real_loss = [nonsat_real_discriminator_loss for _ in range(len(self.resolutions))]
        fake_loss = [nonsat_fake_discriminator_loss for _ in range(len(self.resolutions))]
        losses = real_loss + fake_loss
        self.discriminator_model.compile(
            optimizer=Adam(lr=self.d_lr, beta_1=0.0, beta_2=0.99),
            loss=losses
            )
        self.frozen_discriminator.trainable = fake_labels

        # build generator model
        noise_inputs = [Input((dim, dim, self.img_depth)) for dim in self.resolutions]
        fake_imgs = self.generator(noise_inputs)
        fake_labels = self.frozen_discriminator(fake_imgs)
        g_output = fake_labels + fake_imgs
        self.generator_model = Model(noise_inputs, g_output)
        label_loss = [nonsat_generator_loss for _ in range(len(self.resolutions))]
        mse_loss = ['mse' for _ in range(len(self.resolutions))]
        generator_losses = label_loss + mse_loss
        loss_weights = np.concatenate(
            [np.ones((len(self.resolutions))), self.alpha*np.ones((len(self.resolutions)))]
            )
        self.generator_model.compile(
            optimizer=Adam(lr=self.g_lr, beta_1=0.0, beta_2=0.99),
            loss=generator_losses,
            loss_weight=loss_weights
            )

    ###############################
    ## All our training, etc
    ###############################
    def train(self, epochs, n_batches=16):
        img_generator = ImgGenerator(
            img_dir=self.training_dir,
            batch_size=1,
            n_cpu=self.n_cpu,
            img_dim=self.img_dim_x,
            multiscale=True
            )
        img_batch, _ = img_generator.next()

        noise = [np.random.normal(0, 1, (1, dim, dim, 3)) for dim in self.resolutions]
        for epoch in range(epochs):
            real_batch, _ = img_generator.next()
            d_dummy = [np.ones((1, )) for _ in range(len(self.resolutions)*2)]
            g_dummy = [np.ones((1, )) for _ in range(len(self.resolutions))]
            recon_accum = []
            g_accum = []
            d_accum = []

            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                fake_batch = self.generator.predict(noise)
                d_batch = real_batch + fake_batch
                for img in d_batch:
                    print(img.shape)
                d_loss = self.discriminator_model.train_on_batch(d_batch, d_dummy)
                print("trained d on batch")
                g_true = g_dummy + real_batch
                g_loss = self.generator_model.train_on_batch(noise, g_true)
                print("trained g on batch")
                d_accum.append(d_loss[0])
                g_accum.append(np.mean(g_loss[1:len(self.resolutions)]))
                recon_accum.append(np.mean(g_loss[len(self.resolutions):]))

                pbar.update()
            pbar.close()

            print('{}/{} --> d loss: {:.6f}, g loss: {:.6f}, mse loss: {:.6f}'.format(
                epoch, 
                epochs, 
                np.mean(d_accum),
                np.mean(g_accum),
                np.mean(recon_accum))
                )

            if epoch % self.save_freq == 0:
                # test reconstruction
                test_noise = [np.random.normal(0, 1, (16, dim, dim, 3)) for dim in self.resolutions]
                test_imgs = self.generator.predict(test_noise)
                self.reconstruction_validation(test_imgs, epoch)
                self.save_model_weights(epoch, np.mean(g_accum))
        img_generator.end()

    ###############################
    ## Utilities
    ###############################

    def reconstruction_validation(self, target, n_batch):
        print('Generating Images...')
        if not os.path.isdir(self.validation_dir):
            os.mkdir(self.validation_dir)

        # fill a grid
        reconstructed_imgs = (target+1)*127.5
        grid_dim = int(np.sqrt(reconstructed_imgs.shape[0]))
        img_grid = np.zeros(shape=(self.img_dim_x*grid_dim, 
                                   self.img_dim_y*grid_dim,
                                   self.img_depth))

        positions = itertools.product(range(grid_dim), range(grid_dim))
        for (x_i, y_i), img in zip(positions, reconstructed_imgs):
            x = x_i * self.img_dim_x
            y = y_i * self.img_dim_y
            img_grid[y:y+self.img_dim_y, x:x+self.img_dim_x, :] = img

        savename = os.path.join(self.validation_dir, "{}_sample_img_{}.png".format(self.name, n_batch))
        cv2.imwrite(savename, img_grid.astype(np.uint8)[..., ::-1])

    def save_model_weights(self, epoch, loss):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        g_savename = os.path.join(
            self.checkpoint_dir,
            '{}_g_weights_{}_{:.3f}.h5'.format(self.name, epoch, loss)
            )
        d_savename = os.path.join(
            self.checkpoint_dir,
            '{}_d_weights_{}_{:.3f}.h5'.format(self.name, epoch, loss)
            )
        self.generator.save_weights(g_savename)
        self.discriminator.save_weights(d_savename)
 