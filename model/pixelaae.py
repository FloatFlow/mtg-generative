import cv2
from PIL import Image
import os
import numpy as np
from functools import partial
from tqdm import tqdm
import itertools

from keras.layers import Dense, Reshape, Lambda, Multiply, Add, \
    Activation, UpSampling2D, AveragePooling2D, Input, \
    Concatenate, Flatten, Conv2D, Conv2DTranspose, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import VarianceScaling
from keras.engine.network import Network
import keras.backend as K
import tensorflow as tf

from model.utils import CardGenerator, label_generator, kl_loss
from model.network_blocks import style2_discriminator_block, style2_generator_layer, to_rgb, gated_masked_conv2d
from model.layers import LearnedConstantLatent


class PixelAAE():
    def __init__(
        self,
        lr,
        training_dir,
        validation_dir,
        checkpoint_dir,
        testing_dir
        ):
        # write directories
        self.training_dir = training_dir
        self.validation_dir = validation_dir
        self.checkpoint_dir = checkpoint_dir
        self.testing_dir = testing_dir
        for path in [self.validation_dir,
                     self.checkpoint_dir,
                     self.testing_dir]:
            if not os.path.isdir(path):
                os.makedirs(path)

        # hyperparams
        self.name = 'pixelaae'
        self.kernel_init = 'he_uniform'
        self.img_dim_x = 256
        self.img_dim_y = 256
        self.img_depth = 3
        self.lr = lr
        self.latent_dim = 128
        self.n_classes = 5

        # init models
        self.build_encoder()
        self.build_decoder()
        self.build_style_discriminator()
        self.build_color_discriminator()
        self.build_model()
        print("Model Name: {}".format(self.name))

    ###############################
    ## All our architecture
    ###############################

    def build_encoder(self, ch=16):
        model_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        
        ch = 16
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_uniform'
            )(model_in)
        x = LeakyReLU(0.2)(x)

        while ch < self.latent_dim:
            x = style2_discriminator_block(x, ch)
            ch = ch*2

        while K.int_shape(x)[1] > 4:
            x = style2_discriminator_block(x, ch)

        # 4x4
        x = Conv2D(
            filters=ch,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_uniform'
            )(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(
            filters=ch,
            kernel_size=4,
            padding='valid',
            kernel_initializer='he_uniform'
            )(x)
        x = LeakyReLU(0.2)(x)

        # architecture of tail stem
        x = Flatten()(x)
        x = Dense(self.latent_dim)(x)
        x = LeakyReLU(0.2)(x)
        style = Dense(self.latent_dim)(x)
        
        color = Dense(self.n_classes)(x)
        color = LeakyReLU(0.2)(color)
        color = Dense(self.n_classes)(color)
        color = Activation('sigmoid')(color)
        
        self.encoder = Model(model_in, [style, color])
        print(self.encoder.summary())

    def build_style_discriminator(self):
        style_in = Input(shape=(self.latent_dim, ))
        x = style_in
        for _ in range(4):
            x = Dense(self.latent_dim)(x)
            x = LeakyReLU(0.2)(x)
        x = Dense(1)(x)
        label = Activation('sigmoid')(x)
        self.style_discriminator = Model(style_in, label)
        self.frozen_style_discriminator = Network(style_in, label)
        print(self.style_discriminator.summary())

    def build_color_discriminator(self):
        color_in = Input(shape=(self.n_classes, ))
        x = color_in
        for _ in range(4):
            x = Dense(self.n_classes*4)(x)
            x = LeakyReLU(0.2)(x)
        x = Dense(1)(x)
        label = Activation('sigmoid')(x)
        self.color_discriminator = Model(color_in, label)
        self.frozen_color_discriminator = Network(color_in, label)
        print(self.color_discriminator.summary())

    def build_decoder(self):
        style_in = Input(shape=(self.latent_dim, ))
        class_in = Input(shape=(self.n_classes, ))
        class_embed = Dense(self.n_classes, kernel_initializer='he_uniform')(class_in)
        style = Concatenate()([style_in, class_embed])
        #style = Lambda(
        #    lambda x: x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + K.epsilon())
        #    )(style)
        for _ in range(2):
            style = Dense(units=self.latent_dim, kernel_initializer='he_uniform')(style)
            style = LeakyReLU(0.2)(style)

        ch = self.latent_dim//2
        #x = LearnedConstantLatent()(style_in)
        x = Dense(4*4*ch)(style)
        x = Reshape((4, 4, -1))(x)
        v_stack, h_stack = gated_masked_conv2d(
            v_stack_in=x,
            h_stack_in=x,
            out_dim=ch,
            kernel=5,
            mask='a',
            residual=False,
            context=style,
            use_context=True
            )

        for _ in range(4):
            for _ in range(2):
                v_stack, h_stack = gated_masked_conv2d(
                    v_stack_in=v_stack,
                    h_stack_in=h_stack,
                    out_dim=ch,
                    kernel=5,
                    mask='b',
                    residual=True,
                    context=style,
                    use_context=True
                    )
            v_stack = UpSampling2D(2)(v_stack)
            h_stack = UpSampling2D(2)(h_stack)

        for _ in range(2):
            ch = ch//2
            for _ in range(2):
                v_stack, h_stack = gated_masked_conv2d(
                    v_stack_in=v_stack,
                    h_stack_in=h_stack,
                    out_dim=ch,
                    kernel=5,
                    mask='b',
                    residual=True,
                    context=style,
                    use_context=True
                    )
            v_stack = UpSampling2D(2)(v_stack)
            h_stack = UpSampling2D(2)(h_stack)

        for _ in range(2):
            v_stack, h_stack = gated_masked_conv2d(
                v_stack_in=v_stack,
                h_stack_in=h_stack,
                out_dim=ch,
                kernel=5,
                mask='b',
                residual=True,
                context=style,
                use_context=True
                )
        x = Conv2D(
            filters=3,
            kernel_size=1,
            padding='same'
            )(h_stack)

        model_out = Activation('tanh')(x)

        self.decoder = Model([style_in, class_in], model_out)   
        print(self.decoder.summary()) 

    def build_model(self):
        # build style discriminator
        real_style = Input((self.latent_dim, ))
        fake_style = Input((self.latent_dim, ))
        real_style_label = self.style_discriminator(real_style)
        fake_style_label = self.style_discriminator(fake_style)
        self.style_discriminator_model = Model(
            [real_style, fake_style],
            [real_style_label, fake_style_label],
            name='style_discriminator_model'
            )
        self.style_discriminator_model.compile(
            optimizer=Adam(lr=self.lr),
            loss=['binary_crossentropy', 'binary_crossentropy']
            )
        print(self.style_discriminator_model.summary())

        # build color discriminator
        real_color = Input((self.n_classes, ))
        fake_color = Input((self.n_classes, ))
        real_color_label = self.color_discriminator(real_color)
        fake_color_label = self.color_discriminator(fake_color)
        self.color_discriminator_model = Model(
            [real_color, fake_color],
            [real_color_label, fake_color_label],
            name='color_discriminator_model'
            )
        self.color_discriminator_model.compile(
            optimizer=Adam(lr=self.lr),
            loss=['binary_crossentropy', 'binary_crossentropy']
            )
        print(self.color_discriminator_model.summary())

        # build encoder-decoder
        self.frozen_color_discriminator.trainable = False
        self.frozen_style_discriminator.trainable = False

        img_in = Input((self.img_dim_y, self.img_dim_x, self.img_depth))
        style, color = self.encoder(img_in)
        reconstructed_img = self.decoder([style, color])
        style_label = self.frozen_style_discriminator(style)
        color_label = self.frozen_color_discriminator(color)

        self.aae = Model(img_in, [reconstructed_img, style_label, color_label])
        self.aae.compile(
            optimizer=Adam(lr=self.lr),
            loss=['mse', 'binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[1, 0.25, 0.25]
            )
        print(self.aae.summary())

    ###############################
    ## All our training, etc
    ###############################               

    def train(
        self,
        epochs=100,
        batch_size=16,
        n_cpu=4,
        save_freq=5
        ):
        card_generator = CardGenerator(
            img_dir=self.training_dir,
            batch_size=batch_size,
            n_cpu=n_cpu,
            img_dim=self.img_dim_x
            )
        img_batch, label_batch = card_generator.next()
        self.selected_labels = np.array(label_batch[:16])
        self.selected_samples = np.array(img_batch[:16])
        self.reconstruction_validation(self.selected_samples, -1)

        # test pretesting results
        selected_styles, predicted_labels = self.encoder.predict(self.selected_samples)
        reconstructed_imgs = self.decoder.predict([selected_styles, predicted_labels])
        self.reconstruction_validation(reconstructed_imgs, 0.9)
        # test img generation
        fake_labels = label_generator(batch_size)
        fake_latents = np.random.normal(0, 1, (batch_size, self.latent_dim))
        fake_imgs = self.decoder.predict([fake_latents, fake_labels])
        self.reconstruction_validation(fake_imgs, 0.8)

        n_batches = card_generator.n_batches
        for epoch in range(epochs):
            mse_losses = []
            aae_losses = []
            sd_losses = []
            cd_losses = []

            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                # load batch
                img_batch, real_colors = card_generator.next()
                real_colors = np.clip(real_colors, a_min=0.1, a_max=0.9)
                #real_colors = real_colors + np.random.normal(0, 0.1, real_colors.shape)
                #real_colors = np.clip(real_colors, a_min=0.0, a_max=1.0)
                ones_labels = np.ones(shape=(batch_size, 1))
                zeros_labels = np.zeros(shape=(batch_size, 1))
                real_style = np.random.normal(0, 1, (batch_size, self.latent_dim))
                #real_style = np.random.uniform(-1.0, 1.0, (batch_size, self.latent_dim))

                # train discriminators
                fake_style, fake_colors = self.encoder.predict(img_batch)
                sd_loss, _, _ = self.style_discriminator_model.train_on_batch(
                    [real_style, fake_style],
                    [ones_labels, zeros_labels]
                    )
                cd_loss, _, _ = self.color_discriminator_model.train_on_batch(
                    [real_colors, fake_colors],
                    [ones_labels, zeros_labels]
                    )
                sd_losses.append(sd_loss)
                cd_losses.append(cd_loss)

                # train autoencoder
                _, mse, sa_loss, ca_loss = self.aae.train_on_batch(
                    img_batch,
                    [img_batch, ones_labels, ones_labels]
                    )
                
                mse_losses.append(mse)
                aae_losses.append(sa_loss+ca_loss)

                pbar.update()
            pbar.close()

            print('{}/{} --> mse loss: {}, aae loss: {}, color d loss: {}, style d loss: {}'.format(
                epoch, 
                epochs, 
                np.mean(mse_losses),
                np.mean(aae_losses),
                np.mean(cd_losses),
                np.mean(sd_losses)
                ))

            if epoch % save_freq == 0:
                # test reconstruction
                selected_styles, predicted_labels = self.encoder.predict(self.selected_samples)
                reconstructed_imgs = self.decoder.predict([selected_styles, predicted_labels])
                self.reconstruction_validation(reconstructed_imgs, epoch)
                # test img generation
                fake_labels = label_generator(batch_size)
                fake_latents = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_imgs = self.decoder.predict([fake_latents, fake_labels])
                self.reconstruction_validation(fake_imgs, epoch+0.1)
                self.save_model_weights(epoch, np.mean(mse_losses))
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
        sd_savename = os.path.join(
            self.checkpoint_dir,
            '{}_sd_weights_{}_{:.3f}.h5'.format(self.name, epoch, loss)
            )
        cd_savename = os.path.join(
            self.checkpoint_dir,
            '{}_cd_weights_{}_{:.3f}.h5'.format(self.name, epoch, loss)
            )
        self.decoder.save_weights(decoder_savename)
        self.encoder.save_weights(encoder_savename)
        self.style_discriminator.save_weights(sd_savename)
        self.color_discriminator.save_weights(cd_savename)
 