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
from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.utils import to_categorical
import keras.backend as K
import tensorflow as tf
import tensorflow_probability as tfp


from model.utils import CardGenerator, vq_latent_loss, zq_norm, ze_norm, pixelcnn_accuracy, label_generator, ImgGenerator
from model.network_blocks import gated_masked_conv2d, resblock
from model.layers import VectorQuantizer, AttentionBlock, ScaledDotProductAttention


class CycleVQGAN():
    def __init__(
        self, 
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
        self.name = 'cyclevqgan'
        self.kernel_init = VarianceScaling(np.sqrt(2))
        self.codebook_dim = 64
        self.resblock_dim = 128
        self.k = 64
        self.beta = 0.25
        self.conditional = True
        self.n_classes = 5
        self.build_decoder()
        self.build_model()
        print("Model Name: {}".format(self.name))

    ###############################
    ## All our architecture
    ###############################

    def encoder_pass(self, inputs, ch=32):
        x = Conv2D(
            filters=ch,
            kernel_size=4,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_init
            )(inputs)
        x = Activation('relu')(x) #128x32
        ch *= 2
        x = Conv2D(
            filters=ch,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_init
            )(x)
        x = Activation('relu')(x) #64x64

        ch *= 2
        x = Conv2D(
            filters=ch,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_init
            )(x)
        x = Activation('relu')(x) #32x128

        x = resblock(
            x,
            ch,
            kernel_init=self.kernel_init
            ) #32x128
        x = resblock(
            x,
            ch,
            kernel_init=self.kernel_init
            ) #32x128
        
        features = Conv2D(
            filters=self.codebook_dim,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.kernel_init
            )(x)
        return features

    def build_decoder(self):
        latent_in = Input((32, 32, self.codebook_dim))
        ch = self.codebook_dim*2
        x = Conv2D(
            filters=ch,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=self.kernel_init
            )(latent_in)

        x = resblock(
            x,
            ch,
            kernel_init=self.kernel_init
            ) #32x128
        x = resblock(
            x,
            ch,
            kernel_init=self.kernel_init
            ) #32x128
        
        x = Conv2DTranspose(
            filters=ch,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_init
            )(x)
        x = Activation('relu')(x) #64x128


        ch = ch//2
        x = Conv2DTranspose(
            filters=ch,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_init
            )(x)
        x = Activation('relu')(x) #128x64
        ch = ch//2
        x = Conv2DTranspose(
            filters=ch,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_init
            )(x)
        x = Activation('relu')(x) #256x32

        x = Conv2D(
            filters=3,
            kernel_size=3,
            padding='same',
            kernel_initializer=VarianceScaling(1)
            )(x)
        decoder_out = Activation('tanh')(x)
        self.decoder = Model(latent_in, decoder_out)

    def build_vae(self, tag='encoder'):
        ## Encoder
        encoder_inputs = Input(shape=(self.img_dim_y, self.img_dim_x, self.img_depth))
        z_e = self.encoder_pass(encoder_inputs)

        ## Shared Vector Quantization
        vector_quantizer = VectorQuantizer(self.k, name="vector_quantizer_{}".format(tag))
        codebook_indices = vector_quantizer(z_e)
        self.encoder = Model(
            inputs=encoder_inputs,
            outputs=codebook_indices,
            name='encoder'
            )

        ## Decoder already built
    
        ## VQVAE Model (training)
        sampling_layer = Lambda(lambda x: vector_quantizer.sample(K.cast(x, "int32")), name="sample_from_codebook_{}".format(tag))
        straight_through = Lambda(lambda x : x[1] + tf.stop_gradient(x[0] - x[1]), name="straight_through_estimator_{}".format(tag))

        z_q = sampling_layer(codebook_indices)
        codes = Concatenate(axis=-1)([z_e, z_q])
        straight_through_zq = straight_through([z_q, z_e])
        reconstructed = self.decoder(straight_through_zq)
        #self.vq_vae = Model(inputs=encoder_inputs, outputs=[reconstructed, codes], name='vq-vae')
        vq_vae = Model(inputs=encoder_inputs, outputs=[reconstructed, codes], name='vq-vae_{}'.format(tag))
        return vq_vae
    
        ## VQVAE model (inference)
        #codebook_indices = Input(shape=(32, 32), name='discrete_codes', dtype=tf.int32)
        #z_q = sampling_layer(codebook_indices)
        #generated = self.decoder(z_q)
        #self.vq_vae_sampler = Model(inputs=codebook_indices, outputs=generated, name='vq-vae-sampler')
        
        ## Transition from codebook indices to model (for training the prior later)
        #indices = Input(shape=(32, 32), name='codes_sampler_inputs', dtype='int32')
        #z_q = sampling_layer(indices)
        #self.codes_sampler = Model(
        #    inputs=indices,
        #    outputs=z_q,
        #    name="codes_sampler"
        #    )

    def build_discriminator(self):
        model_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        class_in = Input(shape=(self.n_classes,))
        
        ch = 16
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_uniform'
            )(model_in)
        x = LeakyReLU(0.2)(x)

        while ch < self.z_len:
            x = style2_discriminator_block(x, ch)
            ch = ch*2

        while K.int_shape(x)[1] > 4:
            x = style2_discriminator_block(x, ch)

        # 4x4
        x = MiniBatchStd()(x)
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
        x = Flatten()(x)

        true_fake_eval = Dense(units=1, kernel_initializer='he_uniform')(x)
        class_proba = Dense(units=self.n_classes, kernel_initializer='he_uniform', activation='sigmoid')(x)

        self.discriminator = Model(model_in, [true_fake_eval, class_proba])
        self.frozen_discriminator = Network(model_in, [true_fake_eval, class_proba])
    
    def build_model(self):
        fake_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        real_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        fake_label, fake_class = self.discriminator(fake_in)
        real_label, real_class = self.discriminator(real_in)
        self.discriminator_model = Model(
            [real_in, fake_in],
            [real_label, fake_label, real_class, fake_class])
        self.discriminator_model.compile(
            Adam(self.lr, beta_1=0.0, beta_2=0.999),
            loss=[
                nonsat_real_discriminator_loss,
                nonsat_fake_discriminator_loss,
                'binary_crossentropy',
                'binary_crossentropy'
                #partial(gradient_penalty_loss, averaged_samples=real_in)
                ]
            )
        self.frozen_discriminator.trainable = False

        input_img = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        real_class_in = Input(shape=(self.n_classes, ))
        fake_class_in = Input(shape=(self.n_classes, ))

        real_class_embed = Dense(3, kernel_initializer='he_uniform')(real_class_in)
        real_class_embed = Reshape((1, 1, 3))(real_class_embed)
        fake_class_embed = Dense(3, kernel_initializer='he_uniform')(fake_class_in)
        fake_class_embed = Reshape((1, 1, 3))(fake_class_embed)

        real_x = Concatenate()([input_img, fake_class_embed])
        encoder = self.build_vae(tag='encoder')
        fake_reconstruction, fake_codes = encoder(real_x)

        fake_x = Concatenate()([fake_reconstruction, real_class_embed])
        decoder = self.build_vae(tag='decoder')
        real_reconstruction, real_codes = decoder(fake_x)

        frozen_fake_label, frozen_fake_class = self.frozen_discriminator(fake_reconstruction)

        self.cycle_vae = Model(
            [input_img, real_class_in, fake_class_in],
            [real_reconstruction, real_codes, fake_codes, frozen_fake_label, frozen_fake_class]
            )

        # compile our models
        self.cycle_vae.compile(
            optimizer=Adam(self.lr, beta_1=0.0, beta_2=0.999),
            loss=[
                'mse',
                partial(vq_latent_loss, beta=self.beta),
                partial(vq_latent_loss, beta=self.beta),
                nonsat_generator_loss,
                'binary_crossentropy'
                ],
            #metrics=[zq_norm, ze_norm]
            )
        print(self.cycle_vae.summary())
        print(self.discriminator_model.summary())
        #print("Model Metrics: {}".format(self.vq_vae.metrics_names))

    ###############################
    ## All our training, etc
    ###############################               

    def train(self, epochs):
        card_generator = ImgGenerator(
            img_dir=self.training_dir,
            batch_size=self.batch_size,
            n_cpu=self.n_cpu,
            img_dim=self.img_dim_x
            )

        self.selected_samples, _ = card_generator.next()
        self.reconstruction_target(self.selected_samples, -1)

        n_batches = card_generator.n_batches
        for epoch in range(epochs):
            recon_losses = []
            d_losses = []
            d_label_losses = []
            g_losses = []
            kl_losses = []
            g_label_losses = []

            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                real_images, real_labels = card_generator.next()
                fake_labels = generate_labels(n_samples=self.batch_size, n_classes=self.n_classes)
                kl_dummy = np.zeros((self.batch_size, 32, 32, self.codebook_dim*2))
                dummy = np.ones(shape=(self.batch_size, ))

                fake_imgs, _, _, _, _ = self.cycle_vae.predict([real_imgs, real_labels, fake_labels])
                _, d_real_loss, d_fake_loss, d_real_label_loss, d_fake_label_loss = self.discriminator_model.train_on_batch(
                    [real_imgs, fake_imgs],
                    [dummy, dummy, real_labels, fake_labels]
                    )
                d_loss = d_real_loss + d_fake_loss
                d_label_loss = d_real_label_loss + d_fake_label_loss

                _, recon_loss, real_kl_loss, fake_kl_loss, g_loss, class_loss = self.cycle_vae.train_on_batch(
                    real_batch,
                    [real_batch, kl_dummy, kl_dummy, dummy, real_labels]
                    )
                kl_loss = real_kl_loss + fake_kl_loss
                recon_losses.append(recon_loss)
                kl_losses.append(kl_loss)
                g_losses.append(g_loss)
                g_label_losses.append(class_loss)

                pbar.update()
            pbar.close()

            print('{}/{} --> d loss: {}, g loss: {}, recon loss: {}, kl loss: {}, d label loss: {}, g label loss: {}'.format(
                epoch, 
                epochs, 
                np.mean(d_losses),
                np.mean(g_losses),
                np.mean(recon_losses),
                np.mean(kl_losses),
                np.mean(d_label_losses),
                np.mean(g_label_losses)
                ))

            if epoch % self.save_freq == 0:
                reconstructed_imgs, _, _, _, _ = self.cycle_vae.predict(self.selected_samples)
                self.reconstruction_target(reconstructed_imgs, epoch)
                self.save_model_weights(epoch, np.mean(recon_accum))
        card_generator.end()

    def train_pixelcnn(self, epochs):
        card_generator = CardGenerator(
            img_dir=self.training_dir,
            batch_size=self.batch_size,
            n_cpu=self.n_cpu,
            img_dim=self.img_dim_x
            )
        img_generator = ImgGenerator(
            img_dir=self.training_dir,
            batch_size=self.batch_size,
            n_cpu=self.n_cpu,
            img_dim=self.img_dim_x
            )
        
        #self.generate_samples(-1)

        n_batches = card_generator.n_batches
        for epoch in range(epochs):
            losses = []
            accuracies = []

            self.generate_from_random(-1)
            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                if batch_i % 2 == 0:
                    real_batch, real_labels = card_generator.next()
                else:
                    real_batch, real_labels = img_generator.next()
                codebook_indices = self.encoder.predict(real_batch)
                one_hot = to_categorical(codebook_indices)
                loss, acc = self.pixelcnn.train_on_batch(
                    [one_hot, real_labels],
                    np.expand_dims(codebook_indices, axis=-1)
                    )

                losses.append(loss)
                accuracies.append(acc)
                pbar.update()
            pbar.close()

            print('{}/{} --> sparse bce loss: {}, accuracy: {}'.format(
                epoch, 
                epochs, 
                np.mean(losses),
                np.mean(accuracies))
                )

            if epoch % self.save_freq == 0:
                self.save_model_weights(epoch, np.mean(losses))
                self.save_model_weights_extended(epoch, np.mean(losses))
                self.generate_samples(epoch)


        card_generator.end()

    def sample_from_prior(self, class_labels):
        """sample from the PixelCNN prior, pixel by pixel"""
        X = np.zeros((16, 32, 32, self.k), dtype=np.int32)
        
        pbar = tqdm(total=X.shape[1])
        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                samples  = self.pixelsampler.predict([X, class_labels])
                samples = to_categorical(samples)
                X[:, i, j, :] = samples[:, i, j, :]
            pbar.update()
        pbar.close()
        
        return np.argmax(X, axis=-1)

    def generate_samples(self, epoch):
        print("Generating Samples from Prior...")
        class_labels = label_generator(16)
        indices = self.sample_from_prior(class_labels)
        zq = self.codes_sampler.predict(indices)
        generated = self.decoder.predict(zq, steps=1)
        self.reconstruction_target(generated, epoch)

    def generate_from_random(self, epoch):
        print("Generating Naive Samples...")
        indices = np.random.randint(0, self.k, size=(16, 32, 32))
        zq = self.codes_sampler.predict(indices)
        generated = self.decoder.predict(zq, steps=1)
        self.reconstruction_target(generated, '{}-random'.format(epoch))


    ###############################
    ## Utilities
    ###############################  

    def reconstruction_target(self, target, n_batch):
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
        modelnames = ['encoder', 'decoder', 'codes_sampler', 'vqvae']
        models = [self.encoder, self.decoder, self.codes_sampler, self.vq_vae]
        savenames = ['vqvae_{}_{}.h5'.format(name, epoch) for name, epoch in zip(modelnames, [epoch]*len(modelnames))]
        savenames = [os.path.join(self.checkpoint_dir, sname) for sname in savenames]
        for i, model in enumerate(models):
            model.save_weights(savenames[i])

    def save_model_weights_extended(self, epoch, loss):
        model_names = ["pixelcnn", "pixelsampler"]
        for i, model in enumerate([self.pixelcnn, self.pixelsampler]):
            savename = os.path.join(
                self.checkpoint_dir,
                'vqvae_{}_weights_{}_{:.3f}.h5'.format(model_names[i], epoch, loss)
                )
            model.save_weights(savename)
 