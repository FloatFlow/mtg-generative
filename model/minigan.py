"""
A smaller version of biggan
Seems to work better than stylegan on small, diverse datasets
"""
from tqdm import tqdm
import cv2
from PIL import Image
from keras.utils import plot_model
from functools import partial

from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer
from keras.layers import Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.layers import Concatenate, Embedding, Flatten, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.engine.network import Network

from model.utils import *
from model.layers import *
from model.network_blocks import *

class MiniGAN():
    def __init__(self, 
                 img_dim_x,
                 img_dim_y,
                 img_depth,
                 z_len,
                 n_classes,
                 g_lr,
                 d_lr,
                 batch_size,
                 save_freq,
                 training_dir,
                 validation_dir,
                 checkpoint_dir,
                 testing_dir,
                 n_cpu,
                 n_noise_samples=16):
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.img_depth = img_depth
        self.z_len = z_len
        self.n_noise_samples = n_noise_samples
        self.g_lr = g_lr
        self.d_lr = d_lr
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
        self.n_classes = n_classes
        self.name = 'minigan'
        self.noise_samples = np.concatenate(
            [np.random.normal(0,0.8,size=(self.n_noise_samples, self.z_len)), label_generator(self.n_noise_samples)],
            axis=-1
            )

        self.kernel_init = 'he_normal'
        self.ch = 32
        self.build_generator()
        self.build_discriminator()
        self.build_model()


    ###############################
    ## All our architecture
    ###############################

    '''
    To get # of filters to match:
    G:      D:
    1024    64
    2048    128
    '''
    def build_generator(self):
        model_in = Input(shape=(self.z_len+self.n_classes,))
        x = Dense(4*4*16*self.ch)(model_in)
        x = Reshape((4,4,-1))(x)
        #x = LearnedConstantLatent()(style)

        x = deep_biggan_generator_block(x, model_in, 16*self.ch, upsample=False)
        x = deep_biggan_generator_block(x, model_in, 16*self.ch, upsample=True) #8x256

        #x = deep_biggan_generator_block(x, model_in, 16*self.ch, upsample=False)
        x = deep_biggan_generator_block(x, model_in, 8*self.ch, upsample=True) #16x128

        #x = deep_biggan_generator_block(x, model_in, 8*self.ch, upsample=False)
        x = deep_biggan_generator_block(x, model_in, 8*self.ch, upsample=True) #32x128

        #x = deep_biggan_generator_block(x, model_in, 8*self.ch, upsample=False)
        x = deep_biggan_generator_block(x, model_in, 4*self.ch, upsample=True) #64x64

        x = Attention2SN(4*self.ch)(x)

        #x = deep_biggan_generator_block(x, model_in, 4*self.ch, upsample=False)
        x = deep_biggan_generator_block(x, model_in, 2*self.ch, upsample=True) #128x32

        #x = deep_biggan_generator_block(x, model_in, 2*self.ch, upsample=False)
        x = deep_biggan_generator_block(x, model_in, self.ch, upsample=True) #256x16

        #x = InstanceNormalization()(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        model_out = ConvSN2D(
            filters=3,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer=self.kernel_init,
            activation='tanh'
            )(x)

        self.generator = Model(model_in, model_out)   
        print(self.generator.summary())   


    def build_discriminator(self):
        model_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        class_in = Input(shape=(self.n_classes,))
        x = ConvSN2D(
            filters=self.ch,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer=self.kernel_init
            )(model_in)
        x = deep_biggan_discriminator_block(x, 2*self.ch, downsample=True)
        #x = deep_biggan_discriminator_block(x, 2*self.ch, downsample=False)

        x = deep_biggan_discriminator_block(x, 4*self.ch, downsample=True)
        #x = deep_biggan_discriminator_block(x, 4*self.ch, downsample=False)

        x = Attention2SN(4*self.ch)(x)

        x = deep_biggan_discriminator_block(x, 8*self.ch, downsample=True)
        #x = deep_biggan_discriminator_block(x, 8*self.ch, downsample=False)
        
        x = deep_biggan_discriminator_block(x, 8*self.ch, downsample=True)
        #x = deep_biggan_discriminator_block(x, 8*self.ch, downsample=False)

        x = deep_biggan_discriminator_block(x, 16*self.ch, downsample=True)
        #x = deep_biggan_discriminator_block(x, 16*self.ch, downsample=False)

        x = MiniBatchStd()(x)
        x = deep_biggan_discriminator_block(x, 16*self.ch, downsample=True)
        #x = deep_biggan_discriminator_block(x, 16*self.ch, downsample=False)

        x = Activation('relu')(x)
        x = GlobalSumPooling2D()(x)

        # architecture of tail stem
        out = Dense(1)(x)
        #print('Pooling shape: {}'.format(x.shape))
        y = Dense(1)(class_in)
        #print('Embedding shape: {}'.format(y.shape))
        target_dim = x.shape[-1]
        y = Lambda(lambda x: K.tile(x, (1, target_dim)))(y)
        yh = Multiply()([y, x])
        #yh = Lambda(lambda x: K.sum(x, axis=[1]), output_shape=(self.batch_size,))(yh)
        yh = Lambda(lambda x: K.sum(x, axis=1), output_shape=(1,))(yh)
        model_out = Add()([out, yh])

        self.discriminator = Model([model_in, class_in], model_out)
        self.frozen_discriminator = Network([model_in, class_in], model_out)

        print(self.discriminator.summary())

    def build_model(self):
        d_optimizer = Adam(lr=self.d_lr, beta_1=0.0, beta_2=0.9)
        g_optimizer = Adam(lr=self.g_lr, beta_1=0.0, beta_2=0.9)

        # build complete discriminator
        fake_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        real_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        class_in = Input(shape=(self.n_classes,))
        fake_label = self.discriminator([fake_in, class_in])
        real_label = self.discriminator([real_in, class_in])

        self.discriminator_model = Model(
            [real_in, fake_in, class_in],
            [real_label, fake_label, real_label])
        self.discriminator_model.compile(
            d_optimizer,
            loss=[
                hinge_real_discriminator_loss,
                hinge_fake_discriminator_loss,
                partial(gradient_penalty_loss, averaged_samples=real_in)
                ]
            )

        self.frozen_discriminator.trainable = False

        # build generator model
        z_in = Input(shape=(self.z_len+self.n_classes,))
        class_in = Input(shape=(self.n_classes,))
        fake_img = self.generator(z_in)
        frozen_fake_label = self.frozen_discriminator([fake_img,class_in])

        self.generator_model = Model([z_in, class_in], frozen_fake_label)
        self.generator_model.compile(g_optimizer, hinge_generator_loss)
        
        print(self.discriminator_model.summary())
        print(self.generator_model.summary())

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
        img_generator = ImgGenerator(
            img_dir='agglomerated_images',
            batch_size=self.batch_size,
            n_cpu=self.n_cpu,
            img_dim=self.img_dim_x
            )
        n_batches = card_generator.n_batches*2
        for epoch in range(epochs):
            d_loss_accum = []
            g_loss_accum = []

            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                if batch_i % 2 == 0:
                    real_batch, real_labels = card_generator.next()
                else:
                    real_batch, real_labels = img_generator.next()

                noise = np.random.normal(0, 1, size=(self.batch_size, self.z_len))
                noise_labels = np.concatenate([noise, real_labels], axis=-1)
                dummy = np.ones(shape=(self.batch_size,))

                fake_batch = self.generator.predict(noise_labels)
                
                d_loss = self.discriminator_model.train_on_batch(
                    [real_batch, fake_batch, real_labels],
                    [dummy, dummy, dummy]
                    )
                d_loss_accum.append(d_loss[0])
            
                g_loss = self.generator_model.train_on_batch([noise_labels, real_labels], dummy)
                g_loss_accum.append(g_loss)
                

                pbar.update()
            pbar.close()

            print('{}/{} ----> d_loss: {}, g_loss: {}'.format(
                epoch, 
                epochs, 
                np.mean(d_loss_accum), 
                np.mean(g_loss_accum)
                ))

            if epoch % self.save_freq == 0:
                self.noise_validation(epoch)
                self.save_model_weights(epoch, np.mean(d_loss_accum), np.mean(g_loss_accum))

            card_generator.shuffle()

        card_generator.end()                


    def noise_validation(self, epoch):
        print('Generating Images...')
        if not os.path.isdir(self.validation_dir):
            os.mkdir(self.validation_dir)
        predicted_imgs = self.generator.predict(self.noise_samples)
        predicted_imgs = [((img+1)*127.5).astype(np.uint8) for img in predicted_imgs]

        # fill a grid
        grid_dim = int(np.sqrt(self.n_noise_samples))
        img_grid = np.zeros(shape=(self.img_dim_x*grid_dim, 
                                   self.img_dim_y*grid_dim,
                                   self.img_depth))


        positions = itertools.product(range(grid_dim), range(grid_dim))
        for (x_i, y_i), img in zip(positions, predicted_imgs):
            x = x_i * self.img_dim_x
            y = y_i * self.img_dim_y
            img_grid[y:y+self.img_dim_y, x:x+self.img_dim_x, :] = img

        img_grid = Image.fromarray(img_grid.astype(np.uint8))
        img_grid.save(os.path.join(self.validation_dir, "{}_validation_img_{}.png".format(self.name, epoch)))

    def save_model_weights(self, epoch, d_loss, g_loss):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        generator_savename = os.path.join(self.checkpoint_dir, 'minigan_generator_weights_{}_{:.3f}.h5'.format(epoch, g_loss))
        discriminator_savename = os.path.join(self.checkpoint_dir, 'minigan_discriminator_weights_{}_{:.3f}.h5'.format(epoch, d_loss))
        self.generator.save_weights(generator_savename)
        self.discriminator.save_weights(discriminator_savename)
