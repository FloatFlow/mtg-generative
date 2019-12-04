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
from keras.optimizers import Adam
from keras.initializers import VarianceScaling
import keras.backend as K
import tensorflow as tf

from model.utils import CardGenerator, label_generator, kl_loss
from model.network_blocks import style_discriminator_block, style_generator_block
from model.layers import ReparameterizationTrick


class StyleVAE():
    def __init__(self, 
                 img_dim_x,
                 img_dim_y,
                 img_depth,
                 lr,
                 batch_size,
                 save_freq,
                 training_dir,
                 validation_dir,
                 checkpoint_dir,
                 testing_dir,
                 n_cpu):
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.img_depth = img_depth
        self.lr = lr
        self.batch_size = batch_size
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
        self.name = 'stylevae'
        self.kernel_init = VarianceScaling(np.sqrt(2))
        self.latent_dim = 128
        self.n_classes = 5
        self.build_encoder()
        self.build_decoder()
        self.build_model()
        print("Model Name: {}".format(self.name))

    ###############################
    ## All our architecture
    ###############################

    def build_encoder(self, inputs, ch=16):
        img_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        
        x = Conv2D(filters=ch,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer=self.kernel_init)(img_in)
        x = LeakyReLU(0.2)(x)

        ch *= 2
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init) #128x32
        ch *= 2
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init) #64x64
        ch *= 2
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init) #32x128
        ch *= 2
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init) #16x128
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init) #8x128
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init) #4x128
        x = Flatten()(x)

        # note that latent dim must be divisible by batch size
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        z = ReparameterizationTrick()([z_mean, z_log_var])
        self.encoder = Model(
            img_in,
            [z, z_mean, z_log_var]
            )
        print(self.encoder.summary())

    def build_decoder(self):
        style_in = Input(shape=(self.latent_dim, ))
        label_in = Input(shape=(self.n_classes, ))
        style = Concatenate(axis=-1)([style_in, label_in])
        for _ in range(4):
            style = Dense(self.latent_dim, kernel_initializer=VarianceScaling(1))(style)
            style = LeakyReLU(0.2)(style)

        x = LearnedConstantLatent()(style_in)

        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init, upsample=False) #4x128
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init) #8x128
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init) #16x128
        ch = ch // 2
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init) #32x128
        ch = ch // 2
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init) #64x64
        ch = ch // 2
        x = style_generator_block(x, style, ch,  kernel_init=self.kernel_init) #128x32
        ch = ch // 2
        x = style_generator_block(x, style, ch,  kernel_init=self.kernel_init) #256x16

        x = Conv2D(
            filters=3,
            kernel_size=3,
            padding='same',
            kernel_initializer=VarianceScaling(1)
            )(x)
        model_out = Activation('tanh')(x)
        self.decoder = Model([style_in, label_in], model_out)
        print(self.decoder.summary())

    def build_model(self):
        ## Encoder
        img_in = Input((self.img_dim_y, self.img_dim_x, self.img_depth))
        class_label = Input((self.n_classes, ))
        z, z_mean, z_log_var = self.encoder(img_in)
        reconstructed_img = self.decoder([z, class_label])

        self.vae = Model([img_in, class_label], [reconstructed_img, z])
        self.vae.compile(
            optimizer=Adam(lr=self.lr),
            loss=['mse', partial(kl_loss, z_mean=z_mean, z_log_var=z_log_var)],
            loss_weights=[self.img_dim_x, 1]
            )

    ###############################
    ## All our training, etc
    ###############################               

    def train(self, epochs):
        card_generator = CardGenerator(
            img_dir=self.training_dir,
            batch_size=self.batch_size,
            n_cpu=self.n_cpu,
            img_dim=self.img_dim_x
            )
        img_batch, label_batch = card_generator.next()
        self.selected_labels = np.array(label_batch[:16])
        self.selected_samples = np.array(img_batch[:16])
        self.reconstruction_validation(self.selected_samples, -1)

        n_batches = card_generator.n_batches
        for epoch in range(epochs):
            recon_accum = []
            kl_accum = []

            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                img_batch, label_batch = card_generator.next()
                dummy = np.ones((self.batch_size, self.latent_dim))

                recon_loss, kl_loss = self.vae.train_on_batch(
                    [img_batch, label_batch],
                    [img_batch, dummy]
                    )
                
                recon_accum.append(recon_loss)
                kl_accum.append(kl_loss)

                pbar.update()
            pbar.close()

            print('{}/{} --> recon loss: {}, kl loss: {}'.format(
                epoch, 
                epochs, 
                np.mean(recon_accum),
                np.mean(kl_accum))
                )

            if epoch % self.save_freq == 0:
                # test reconstruction
                reconstructed_imgs, _ = self.vae.predict([self.selected_samples, self.selected_labels])
                self.reconstruction_validation(reconstructed_imgs, epoch)
                # test img generation
                fake_labels = label_generator(self.batch_size)
                fake_latents = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                fake_imgs, _ = self.vae.predict([fake_latents, fake_labels])
                self.reconstruction_validation(fake_imgs, epoch+0.1)
                self.save_model_weights(epoch, np.mean(recon_accum))
        card_generator.end()

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
        decoder_savename = os.path.join(
            self.checkpoint_dir,
            '{}_decoder_weights_{}_{:.3f}.h5'.format(self.name, epoch, loss)
            )
        encoder_savename = os.path.join(
            self.checkpoint_dir,
            '{}_encoder_weights_{}_{:.3f}.h5'.format(self.name, epoch, loss)
            )
        self.decoder.save_weights(decoder_savename)
        self.encoder.save_weights(encoder_savename)
 