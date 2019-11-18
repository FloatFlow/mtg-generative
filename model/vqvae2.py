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
import keras.backend as K
import tensorflow as tf
import tensorflow_probability as tfp

from model.utils import CardGenerator, vq_latent_loss, zq_norm, ze_norm, pixelcnn_accuracy, label_generator
from model.network_blocks import resblock_decoder, resblock_encoder, intro_pixelcnn_layer, pixelcnn_layer
from model.layers import VectorQuantizer


class VQVAE2():
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
        self.name = 'vqvae2'
        self.kernel_init = VarianceScaling(np.sqrt(2))
        self.latent_dim = 64
        self.k = 256
        self.beta = 0.25
        self.conditional = True
        self.build_decoder()
        self.build_model()
        print("Model Name: {}".format(self.name))

    ###############################
    ## All our architecture
    ###############################

    def encoder_pass(self, inputs, ch=16):
        x = resblock_encoder(
                inputs,
                ch,
                downsample=False,
                kernel_init=self.kernel_init
                ) #256x16

        ch *= 2
        x = resblock_encoder(
            x,
            ch,
            kernel_init=self.kernel_init
            ) #128x32

        ch *= 2
        x = resblock_encoder(
            x,
            ch,
            kernel_init=self.kernel_init
            ) #64x64
        feat64 = Conv2D(
            filters=self.latent_dim,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.kernel_init
            )(x)

        ch *= 2
        x = resblock_encoder(
            x,
            ch,
            kernel_init=self.kernel_init
            ) #32x128
        feat32 = Conv2D(
            filters=self.latent_dim,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.kernel_init
            )(x)
        return (feat64, feat32)

    def build_decoder(self):
        latent_in32 = Input((32, 32, self.latent_dim))
        ch = self.latent_dim
        x = resblock_decoder(
            latent_in32,
            ch,
            kernel_init=self.kernel_init
            )
        
        latent_in64 = Input((64, 64, self.latent_dim))
        x = Concatenate(axis=-1)([x, latent_in64])
        x = resblock_decoder(
            x,
            ch,
            upsample=False,
            kernel_init=self.kernel_init
            )

        ch = ch//2
        x = resblock_decoder(
            x,
            ch,
            kernel_init=self.kernel_init
            )
        ch = ch//2
        x = resblock_decoder(
            x,
            ch,
            kernel_init=self.kernel_init
            )

        x = Conv2D(
            filters=3,
            kernel_size=3,
            padding='same',
            kernel_initializer=VarianceScaling(1)
            )(x)
        decoder_out = Activation('tanh')(x)
        self.decoder = Model([latent_in32, latent_in64], decoder_out)

    def build_model(self):
        ## Encoder
        encoder_inputs = Input(shape=(self.img_dim_y, self.img_dim_x, self.img_depth))
        z_e64, z_e32 = self.encoder_pass(encoder_inputs)

        ## Shared Vector Quantization
        vector_quantizer = VectorQuantizer(self.k, name="vector_quantizer")
        codebook_indices32 = vector_quantizer(z_e32)
        codebook_indices64 = vector_quantizer(z_e64)
        self.encoder = Model(
            inputs=encoder_inputs,
            outputs=[codebook_indices32, codebook_indices64],
            name='encoder'
            )

        ## Decoder already built
    
        ## VQVAE Model (training)
        sampling_layer = Lambda(lambda x: vector_quantizer.sample(K.cast(x, "int32")), name="sample_from_codebook")
        straight_through = Lambda(lambda x : x[1] + tf.stop_gradient(x[0] - x[1]), name="straight_through_estimator")

        z_q32 = sampling_layer(codebook_indices32)
        z_q64 = sampling_layer(codebook_indices64)
        codes32 = Concatenate(axis=-1)([z_e32, z_q32])
        codes64 = Concatenate(axis=-1)([z_e64, z_q64])
        straight_through_zq32 = straight_through([z_q32, z_e32])
        straight_through_zq64 = straight_through([z_q64, z_e64])
        reconstructed = self.decoder([straight_through_zq32, straight_through_zq64])
        self.vq_vae = Model(inputs=encoder_inputs, outputs=[reconstructed, codes32, codes64], name='vq-vae')
    
        ## VQVAE model (inference)
        codebook_indices_32 = Input(shape=(32, 32), name='discrete_codes32', dtype=tf.int32)
        codebook_indices_64 = Input(shape=(64, 64), name='discrete_codes64', dtype=tf.int32)
        z_q_32 = sampling_layer(codebook_indices_32)
        z_q_64 = sampling_layer(codebook_indices_64)
        generated = self.decoder([z_q_32, z_q_64])
        self.vq_vae_sampler = Model(inputs=[codebook_indices_32, codebook_indices_64], outputs=generated, name='vq-vae-sampler')
        
        ## Transition from codebook indices to model (for training the prior later)
        indices_32 = Input(shape=(32, 32), name='codes_sampler_inputs', dtype='int32')
        indices_64 = Input(shape=(64, 64), name='codes_sampler_inputs64', dtype='int32')
        z_q_32 = sampling_layer(indices_32)
        z_q_64 = sampling_layer(indices_64)
        self.codes_sampler_32 = Model(
            inputs=indices_32,
            outputs=z_q_32,
            name="codes_sampler32"
            )
        self.codes_sampler_64 = Model(
            inputs=indices_64,
            outputs=z_q_64,
            name="codes_sampler64"
            )
        
        ## Getter to easily access the codebook for vizualisation
        #indices = Input(shape=(), dtype='int32')
        #vq_samples = Lambda(lambda x: vector_quantizer.sample(K.cast(x[:, None, None], "int32")))(indices)
        #self.vector_model = Model(inputs=indices, outputs=vq_samples, name='get_codebook')

        # compile our models
        opt = Adam(self.lr, beta_1=0.0, beta_2=0.999)
        self.vq_vae.compile(
            optimizer=opt,
            loss=['mse', partial(vq_latent_loss, beta=self.beta), partial(vq_latent_loss, beta=self.beta)],
            metrics=[zq_norm, ze_norm]
            )
        print(self.vq_vae.summary())
        print("Model Metrics: {}".format(self.vq_vae.metrics_names))
    
    def get_vq_vae_codebook(self):
        codebook = vector_model.predict(np.arange(k))
        codebook = np.reshape(codebook, (k, d))
        return codebook

    def build_pixelcnn(self, input_shape=(32, 32), n_layers=10, conditional=False):
        pixelcnn_prior_inputs = Input(input_shape)
        if input_shape[0] == 32:
            z_q = self.codes_sampler_32(pixelcnn_prior_inputs)
        else:
            z_q = self.codes_sampler_64(pixelcnn_prior_inputs)
        if conditional:
            h = Input((5, ))
        else:
            h = None
        v_stack, h_stack = intro_pixelcnn_layer(
            z_q,
            filter_size=(7, 7),
            n_filters=self.latent_dim,
            h=h
            )
        for _ in range(n_layers):
            v_stack, h_stack = pixelcnn_layer(
                v_stack,
                h_stack,
                h=h,
                n_filters=self.latent_dim)
        x = Conv2D(
            filters=self.latent_dim,
            kernel_size=1,
            padding='same',
            kernel_initializer=VarianceScaling(1)
            )(h_stack)
        autoregression = Conv2D(
            filters=self.k,
            kernel_size=1,
            padding='same',
            kernel_initializer=VarianceScaling(1)
            )(x)

        # Distribution to sample from the pixelcnn
        sampled = Lambda(lambda x: tfp.distributions.Categorical(logits=x).sample())(autoregression)
        if conditional:
            return (
                Model([pixelcnn_prior_inputs, h], autoregression),
                Model([pixelcnn_prior_inputs, h], sampled, name='pixelcnn-prior-sampler')
                )
        else:
            return (
                Model(pixelcnn_prior_inputs, autoregression),
                Model(pixelcnn_prior_inputs, sampled)
                )

    def build_pixelcnns_and_samplers(self, n_layers=10):
        self.pixelcnn32, self.pixelsampler32 = self.build_pixelcnn(
            input_shape=(32, 32),
            n_layers=n_layers,
            conditional=self.conditional
            )
        #with open('pixelcnn32_architecture.txt', 'w') as f:
        #    self.pixelcnn32.summary(print_fn=lambda x: f.write(x + '\n'))
        
        self.pixelcnn64, self.pixelsampler64 = self.build_pixelcnn(
            input_shape=(64, 64),
            n_layers=n_layers,
            conditional=self.conditional
            )

    def compile_pixelcnns(self):
        self.pixelcnn32.compile(
            optimizer=Adam(self.lr, beta_1=0.0, beta_2=0.999),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[pixelcnn_accuracy]
            )
        self.pixelcnn64.compile(
            optimizer=Adam(self.lr, beta_1=0.0, beta_2=0.999),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[pixelcnn_accuracy]
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
        real_batch, real_labels = card_generator.next()
        self.selected_samples = np.array(real_batch[:16])
        self.reconstruction_target(self.selected_samples, -1)
        self.reconstruction_validation(-1)
        n_batches = card_generator.n_batches
        for epoch in range(epochs):
            recon_accum = []
            kl_accum = []
            vq_accum = []
            ve_accum = []

            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                real_batch, real_labels = card_generator.next()
                dummy32 = np.zeros((self.batch_size, 32, 32, self.latent_dim*2))
                dummy64 = np.zeros((self.batch_size, 64, 64, self.latent_dim*2))

                _, recon_loss, kl_loss32, kl_loss64, _, _, vqnorm32, venorm32, vqnorm64, venorm64 = self.vq_vae.train_on_batch(
                    real_batch,
                    [real_batch, dummy32, dummy64]
                    )
                recon_accum.append(recon_loss)
                kl_accum.append(kl_loss32+kl_loss64)
                vq_accum.append(vqnorm32+vqnorm64)
                ve_accum.append(venorm32+venorm64)

                pbar.update()
            pbar.close()

            print('{}/{} --> recon loss: {}, kl loss: {}, z_q norm: {}, z_e norm: {}'.format(
                epoch, 
                epochs, 
                np.mean(recon_accum),
                np.mean(kl_accum),
                np.mean(vq_accum),
                np.mean(ve_accum))
                )

            if epoch % self.save_freq == 0:
                self.reconstruction_validation(epoch)
                self.save_model_weights(epoch, np.mean(recon_accum))
        card_generator.end()

    def train_pixelcnn(self, epochs):

        self.compile_pixelcnns()

        card_generator = CardGenerator(
            img_dir=self.training_dir,
            batch_size=self.batch_size,
            n_cpu=self.n_cpu,
            img_dim=self.img_dim_x
            )
        
        real_batch, real_labels = card_generator.next()

        n_batches = card_generator.n_batches
        for epoch in range(epochs):
            losses = []
            accuracies = []

            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                real_batch, real_labels = card_generator.next()
                codebook_indices32, codebook_indices64 = self.encoder.predict(real_batch)

                if self.conditional:
                    loss32, acc32 = self.pixelcnn32.train_on_batch(
                        [codebook_indices32, real_labels],
                        np.expand_dims(codebook_indices32, axis=-1)
                        )
                    loss64, acc64 = self.pixelcnn64.train_on_batch(
                        [codebook_indices64, real_labels],
                        np.expand_dims(codebook_indices64, axis=-1)
                        )
                else:
                    loss32, acc32 = self.pixelcnn32.train_on_batch(
                        codebook_indices32,
                        np.expand_dims(codebook_indices32, axis=-1)
                        )
                    loss64, acc64 = self.pixelcnn64.train_on_batch(
                        codebook_indices64,
                        np.expand_dims(codebook_indices64, axis=-1)
                        )

                losses.append(np.mean([loss32, loss64]))
                accuracies.append(np.mean([acc32, acc64]))

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

        card_generator.end()

    def sample_from_prior(self, prior_model, class_labels, shape):
        """sample from the PixelCNN prior, pixel by pixel"""
        X = np.zeros(shape, dtype=np.int32)
        pbar = tqdm(total=X.shape[1])
        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                sampled = prior_model.predict([X, class_labels])
                X[:, i, j] = sampled[:, i, j]
            pbar.update()
        pbar.close()
        return X

    def generate_samples(self, n_batches):
        pbar = tqdm(total=n_batches)
        for i in range(n_batches):
            class_labels = label_generator(16)
            indices32 = self.sample_from_prior(self.pixelsampler32, class_labels, (16, 32, 32))
            indices64 = self.sample_from_prior(self.pixelsampler64, class_labels, (16, 64, 64))
            zq32 = self.codes_sampler32(indices32)
            zq64 = self.codes_sampler64(indices64)
            generated = self.decoder.predict([zq32, zq64])
            self.reconstruction_target(generated, i)
            pbar.update()
        pbar.close()

    ###############################
    ## Utilities
    ###############################  

    def reconstruction_validation(self, epoch):
        print('Generating Images...')
        if not os.path.isdir(self.validation_dir):
            os.mkdir(self.validation_dir)
        reconstructed_imgs, _, _ = self.vq_vae.predict(self.selected_samples)
        reconstructed_imgs = ((np.array(reconstructed_imgs)+1)*127.5).astype(np.uint8)

        # fill a grid
        grid_dim = int(np.sqrt(reconstructed_imgs.shape[0]))
        img_grid = np.zeros(shape=(self.img_dim_x*grid_dim, 
                                   self.img_dim_y*grid_dim,
                                   self.img_depth))

        positions = itertools.product(range(grid_dim), range(grid_dim))
        for (x_i, y_i), img in zip(positions, reconstructed_imgs):
            x = x_i * self.img_dim_x
            y = y_i * self.img_dim_y
            img_grid[y:y+self.img_dim_y, x:x+self.img_dim_x, :] = img

        savename = os.path.join(self.validation_dir, "{}_validation_img_{}.png".format(self.name, epoch))
        cv2.imwrite(savename, img_grid.astype(np.uint8)[..., ::-1])

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
        decoder_savename = os.path.join(self.checkpoint_dir, 'vqvae_decoder_weights_{}_{:.3f}.h5'.format(epoch, loss))
        encoder_savename = os.path.join(self.checkpoint_dir, 'vqvae_encoder_weights_{}_{:.3f}.h5'.format(epoch, loss))
        self.decoder.save_weights(decoder_savename)
        self.encoder.save_weights(encoder_savename)

    def save_model_weights_extended(self, epoch, loss):
        model_names = ["pixelcnn32", "pixelsampler32", "pixelcnn64", "pixelsampler64"]
        for i, model in enumerate([self.pixelcnn32, self.pixelsampler32, self.pixelcnn64, self.pixelsampler64]):
            savename = os.path.join(
                self.checkpoint_dir,
                'vqvae_{}_weights_{}_{:.3f}.h5'.format(model_names[i], epoch, loss)
                )
            model.save_weights(savename)
 