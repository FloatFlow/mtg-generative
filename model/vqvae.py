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


class VQVAE():
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
        self.codebook_dim = 64
        self.resblock_dim = 128
        self.k = 64
        self.beta = 0.25
        self.conditional = True
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

    def build_model(self):
        ## Encoder
        encoder_inputs = Input(shape=(self.img_dim_y, self.img_dim_x, self.img_depth))
        z_e = self.encoder_pass(encoder_inputs)

        ## Shared Vector Quantization
        vector_quantizer = VectorQuantizer(self.k, name="vector_quantizer")
        codebook_indices = vector_quantizer(z_e)
        self.encoder = Model(
            inputs=encoder_inputs,
            outputs=codebook_indices,
            name='encoder'
            )

        ## Decoder already built
    
        ## VQVAE Model (training)
        sampling_layer = Lambda(lambda x: vector_quantizer.sample(K.cast(x, "int32")), name="sample_from_codebook")
        straight_through = Lambda(lambda x : x[1] + tf.stop_gradient(x[0] - x[1]), name="straight_through_estimator")

        z_q = sampling_layer(codebook_indices)
        codes = Concatenate(axis=-1)([z_e, z_q])
        straight_through_zq = straight_through([z_q, z_e])
        reconstructed = self.decoder(straight_through_zq)
        self.vq_vae = Model(inputs=encoder_inputs, outputs=[reconstructed, codes], name='vq-vae')
    
        ## VQVAE model (inference)
        #codebook_indices = Input(shape=(32, 32), name='discrete_codes', dtype=tf.int32)
        #z_q = sampling_layer(codebook_indices)
        #generated = self.decoder(z_q)
        #self.vq_vae_sampler = Model(inputs=codebook_indices, outputs=generated, name='vq-vae-sampler')
        
        ## Transition from codebook indices to model (for training the prior later)
        indices = Input(shape=(32, 32), name='codes_sampler_inputs', dtype='int32')
        z_q = sampling_layer(indices)
        self.codes_sampler = Model(
            inputs=indices,
            outputs=z_q,
            name="codes_sampler"
            )
        
        ## Getter to easily access the codebook for vizualisation
        #indices = Input(shape=(), dtype='int32')
        #vq_samples = Lambda(lambda x: vector_quantizer.sample(K.cast(x[:, None, None], "int32")))(indices)
        #self.vector_model = Model(inputs=indices, outputs=vq_samples, name='get_codebook')

        # compile our models
        opt = Adam(self.lr, beta_1=0.0, beta_2=0.999)
        self.vq_vae.compile(
            optimizer=opt,
            loss=['mse', partial(vq_latent_loss, beta=self.beta)],
            metrics=[zq_norm, ze_norm]
            )
        print(self.encoder.summary())
        print(self.decoder.summary())
        print("Model Metrics: {}".format(self.vq_vae.metrics_names))
    
    def get_vq_vae_codebook(self):
        codebook = vector_model.predict(np.arange(k))
        codebook = np.reshape(codebook, (k, d))
        return codebook

    def pixelcnn_pass(self, x, h, n_layers=20, attention=False):
        v_stack, h_stack = gated_masked_conv2d(
            v_stack_in=x,
            h_stack_in=x,
            out_dim=self.k,
            kernel=5,
            mask='a',
            residual=False,
            context=h,
            use_context=True
            )

        for _ in range(n_layers//4):
            for _ in range(4):
                v_stack, h_stack = gated_masked_conv2d(
                    v_stack_in=v_stack,
                    h_stack_in=h_stack,
                    out_dim=self.k,
                    kernel=5,
                    mask='b',
                    residual=True,
                    context=h,
                    use_context=True
                    )
            if attention:
                h_stack = ScaledDotProductAttention()(h_stack)

        x = Conv2D(
            filters=self.k,
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
        return autoregression

    def build_pixelcnn(self, n_layers=20):
        # pixelcnn
        pixelcnn_prior_inputs = Input((32, 32, self.k))
        class_labels = Input((5, ))
        prior = self.pixelcnn_pass(pixelcnn_prior_inputs, h=class_labels, n_layers=n_layers, attention=True)
        sampled = Lambda(lambda x: tfp.distributions.Categorical(logits=x).sample())(prior)

        self.pixelcnn = Model(
            [pixelcnn_prior_inputs, class_labels],
            prior
            )
        self.pixelsampler = Model(
            [pixelcnn_prior_inputs, class_labels],
            sampled
            )

        self.pixelcnn.compile(
            optimizer=Adam(self.lr),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[pixelcnn_accuracy]
            )
        print(self.pixelcnn.summary())
        print("Model Metrics: {}".format(self.pixelcnn.metrics_names))

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
            recon_accum = []
            kl_accum = []
            vq_accum = []
            ve_accum = []

            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                real_batch, _ = card_generator.next()
                dummy = np.zeros((self.batch_size, 32, 32, self.codebook_dim*2))

                _, recon_loss, kl_loss, _, _, vqnorm, venorm = self.vq_vae.train_on_batch(
                    real_batch,
                    [real_batch, dummy]
                    )
                recon_accum.append(recon_loss)
                kl_accum.append(kl_loss)
                vq_accum.append(vqnorm)
                ve_accum.append(venorm)

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
                reconstructed_imgs, _ = self.vq_vae.predict(self.selected_samples)
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
 