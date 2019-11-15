from model.utils import *
from model.layers import *
from model.network_blocks import *
from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer
from keras.layers import Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.layers import Concatenate, Embedding, Flatten, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.engine.network import Network
from tqdm import tqdm
import cv2
from PIL import Image
from keras.utils import plot_model
from functools import partial


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
        self.name = 'vqvae'
        self.kernel_init = VarianceScaling(np.sqrt(2))
        self.latent_dim = 256
        self.activation = 'leaky'
        self.k = 128

    ###############################
    ## All our architecture
    ###############################

    def encoder_pass(self, inputs, ch=16):
        x = style_discriminator_block(
            inputs,
            ch,
            activation=self.activation,
            upsample=False,
            kernel_init=self.kernel_init
            )

        while ch < self.latent_dim:
            ch *= 2
            x = style_discriminator_block(
                x,
                ch,
                activation=self.activation,
                kernel_init=self.kernel_init
                )
        while K.int_shape(x)[1] > 4:
            x = style_discriminator_block(
                x,
                ch,
                activation=self.activation,
                kernel_init=self.kernel_init
                )
        x = Conv2D(
            filters=self.latent_dim,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.kernel_init
            )(x)
        return x

    def build_decoder(self, ch=256):
        latent_in = Input((4, 4, self.latent_dim))
        x = style_decoder_block(
            latent_in,
            ch,
            activation=self.activation,
            kernel_init=self.kernel_init
            )
        for _ in range(2):
            x = style_decoder_block(
                x,
                ch,
                activation=self.activation,
                kernel_init=self.kernel_init
                )
        for _ in range(5):
            ch = ch//2
            x = style_decoder_block(
                x,
                ch,
                activation=self.activation,
                kernel_init=self.kernel_init
                )
        x = Conv2D(
            filters=3,
            kernel_size=3,
            padding='same',
            kernel_initializer=VarianceScaling(1)
            )
        decoder_out = Activation('tanh')(x)
        self.decoder = Model(latent_in, decoder_out)

    def build_model(self):
        ## Encoder
        encoder_inputs = Input(shape=(self.img_dim, self.img_dim, self.img_depth))
        z_e = self.encoder_pass(encoder_inputs, self.latent_dim, num_layers=num_layers)
        SIZE = int(z_e.get_shape()[1])

        ## Vector Quantization
        vector_quantizer = VectorQuantizer(self.k, name="vector_quantizer")
        codebook_indices = vector_quantizer(z_e)
        encoder = K.Model(inputs=encoder_inputs, outputs=codebook_indices, name='encoder')

        ## Decoder already built
    
        ## VQVAE Model (training)
        sampling_layer = Lambda(lambda x: vector_quantizer.sample(K.cast(x, "int32")), name="sample_from_codebook")
        z_q = sampling_layer(codebook_indices)
        codes = Concatenate(axis=-1)([z_e, z_q])
        straight_through = Lambda(lambda x : x[1] + tf.stop_gradient(x[0] - x[1]), name="straight_through_estimator")
        straight_through_zq = straight_through([z_q, z_e])
        reconstructed = self.decoder(straight_through_zq)
        self.vq_vae = Model(inputs=encoder_inputs, outputs=[reconstructed, codes], name='vq-vae')
    
        ## VQVAE model (inference)
        codebook_indices = Input(shape=(SIZE, SIZE), name='discrete_codes', dtype=tf.int32)
        z_q = sampling_layer(codebook_indices)
        generated = self.decoder(z_q)
        self.vq_vae_sampler = Model(inputs=codebook_indices, outputs=generated, name='vq-vae-sampler')
        
        ## Transition from codebook indices to model (for training the prior later)
        indices = Input(shape=(SIZE, SIZE), name='codes_sampler_inputs', dtype='int32')
        z_q = sampling_layer(indices)
        self.codes_sampler = Model(inputs=indices, outputs=z_q, name="codes_sampler")
        
        ## Getter to easily access the codebook for vizualisation
        indices = Input(shape=(), dtype='int32')
        self.vector_model = Model(inputs=indices, outputs=vector_quantizer.sample(indices[:, None, None]), name='get_codebook')

        # compile our models
        opt = ADAM(self.lr, beta1=0.0, beta2=0.999)
        self.vq_vae.compile(
            optimizer=opt,
            loss=['mse', partial(vq_latent_loss, beta=self.beta)],
            metrics=[zq_norm, ze_norm]
            )
    
    def get_vq_vae_codebook(self):
        codebook = vector_model.predict(np.arange(k))
        codebook = np.reshape(codebook, (k, d))
        return codebook

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
        self.selected_samples = real_batch[:16]
            
        n_batches = card_generator.n_batches
        for epoch in range(epochs):
            recon_accum = []
            kl_accum = []
            vq_accum = []
            ve_accum = []

            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                real_batch, real_labels = card_generator.next()
                dummy = np.zeros((batch_size, 4, 4, self.latent_dim*2))

                _, recon_loss, kl_loss, _, _, vqnorm, venorm, _ = self.vq_vae.train_on_batch(
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
                self.reconstruction_validation(epoch)
                self.save_model_weights(epoch, np.mean(loss_accum))

            card_generator.shuffle()
        card_generator.end()

    def reconstruction_validation(self, epoch):
        print('Generating Images...')
        if not os.path.isdir(self.validation_dir):
            os.mkdir(self.validation_dir)
        reconstructed_imgs = self.vq_vae.predict(self.selected_samples)
        reconstructed_imgs = [((img+1)*127.5).astype(np.uint8) for img in reconstructed_imgs]

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

        img_grid = Image.fromarray(img_grid.astype(np.uint8))
        img_grid.save(os.path.join(self.validation_dir, "validation_img_{}.png".format(epoch)))

    def save_model_weights(self, epoch, loss):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        decoder_savename = os.path.join(self.checkpoint_dir, 'vqvae_decoder_weights_{}_{:.3f}.h5'.format(epoch, loss))
        encoder_savename = os.path.join(self.checkpoint_dir, 'vqvae_encoder_weights_{}_{:.3f}.h5'.format(epoch, loss))
        self.decoder.save_weights(decoder_savename)
        self.encoder.save_weights(encoder_savename)
