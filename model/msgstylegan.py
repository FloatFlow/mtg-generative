import os
from tqdm import tqdm
from functools import partial
import numpy as np
import itertools
from PIL import Image

from keras.layers import Dense, Reshape, Lambda, Multiply, Add, \
    Activation, Input, Concatenate, LeakyReLU, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import VarianceScaling
import keras.backend as K
from keras.engine.network import Network

from model.utils import nonsat_generator_loss, nonsat_real_discriminator_loss, nonsat_fake_discriminator_loss, \
    ImgGenerator, CardGenerator, label_generator, gradient_penalty_loss, hinge_real_discriminator_loss, \
    hinge_fake_discriminator_loss, hinge_generator_loss
from model.layers import LatentPixelNormalization, LearnedConstantLatent, MiniBatchStd
from model.network_blocks import style_generator_block, style_discriminator_block


class MSGStyleGAN():
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
        self.name = 'msgstylegan'
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.img_depth = img_depth
        self.latent_dim = z_len
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
        
        self.style_samples = np.random.normal(0, 0.8, size=(self.n_noise_samples, self.latent_dim))
        self.label_samples = label_generator(self.n_noise_samples)
        self.model_name = 'stylegan'
        self.gp_weight = 10 # is really gamma = 5 due to definition
        self.kernel_init = VarianceScaling(scale=np.sqrt(2))
        self.latent_dim = 256
        self.build_generator()
        self.build_discriminator()
        self.build_model()

    ###############################
    ## All our architecture
    ###############################
    def build_generator(self):
        ch = self.latent_dim
        style_in = Input(shape=(self.latent_dim, ))
        label_in = Input(shape=(self.n_classes, ))
        style = LatentPixelNormalization()(style_in)
        label_embed = Dense(5, kernel_initializer=VarianceScaling(1))(label_in)
        style = Concatenate(axis=-1)([style, label_embed])
        for _ in range(4):
            style = Dense(self.latent_dim, kernel_initializer=VarianceScaling(1))(style)
            style = LeakyReLU(0.2)(style)

        x = LearnedConstantLatent()(style_in)
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init, upsample=False)
        fourbyout = Conv2D(
            filters=3,
            kernel_size=1,
            padding='same',
            activation='tanh'
            )(x) #4x256

        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init)
        eightbyout = Conv2D(
            filters=3,
            kernel_size=1,
            padding='same',
            activation='tanh'
            )(x) #8x256
        
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init)
        sixteenbyout = Conv2D(
            filters=3,
            kernel_size=1,
            padding='same',
            activation='tanh'
            )(x) #16x256
        
        ch = ch // 2
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init)
        thirtytwobyout = Conv2D(
            filters=3,
            kernel_size=1,
            padding='same',
            activation='tanh'
            )(x) #32x128

        ch = ch // 2
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init)
        sixtyfourbyout = Conv2D(
            filters=3,
            kernel_size=1,
            padding='same',
            activation='tanh'
            )(x) #64x64

        ch = ch // 2
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init)
        onetwentyeightbyout = Conv2D(
            filters=3,
            kernel_size=1,
            padding='same',
            activation='tanh'
            )(x) #128x32


        ch = ch // 2
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init)
        full_resolution = Conv2D(
            filters=3,
            kernel_size=1,
            padding='same',
            kernel_initializer=VarianceScaling(scale=1),
            activation='tanh'
            )(x)

        self.generator = Model(
            [style_in, label_in],
            [full_resolution, onetwentyeightbyout, sixtyfourbyout, thirtytwobyout, sixteenbyout, eightbyout, fourbyout]
            )   
        print(self.generator.summary())

    def build_discriminator(self, ch=16, kernel_init='he_normal'):
        full_resolution_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        class_in = Input(shape=(self.n_classes, ))
        
        x = Conv2D(
            filters=ch,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.kernel_init
            )(full_resolution_in)
        x = LeakyReLU(0.2)(x)

        ch *= 2
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init)
        onetwentyeightin = Input(shape=(self.img_dim_y//2, self.img_dim_x//2, self.img_depth))
        x = Concatenate()([x, onetwentyeightin])
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer=self.kernel_init,
            padding='same'
            )(x) #128x32

        ch *= 2
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init)
        sixtyfourin = Input(shape=(self.img_dim_y//4, self.img_dim_x//4, self.img_depth))
        x = Concatenate()([x, sixtyfourin])
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer=self.kernel_init,
            padding='same'
            )(x) #64x64

        ch *= 2
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init)
        thirtytwoin = Input(shape=(self.img_dim_y//8, self.img_dim_x//8, self.img_depth))
        x = Concatenate()([x, thirtytwoin])
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer=self.kernel_init,
            padding='same'
            )(x) #32x128

        ch *= 2
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init)
        sixteenin = Input(shape=(self.img_dim_y//16, self.img_dim_x//16, self.img_depth))
        x = Concatenate()([x, sixteenin])
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer=self.kernel_init,
            padding='same'
            )(x) #16x256

        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init)
        eightin = Input(shape=(self.img_dim_y//32, self.img_dim_x//32, self.img_depth))
        x = Concatenate()([x, eightin])
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer=self.kernel_init,
            padding='same'
            )(x) #16x256

        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init)
        fourin = Input(shape=(self.img_dim_y//64, self.img_dim_x//64, self.img_depth))
        x = Concatenate()([x, fourin])
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer=self.kernel_init,
            padding='same'
            )(x) #16x256

        x = MiniBatchStd()(x)
        x = Conv2D(
            filters=ch,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.kernel_init
            )(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(
            filters=ch,
            kernel_size=4,
            padding='valid',
            kernel_initializer=self.kernel_init
            )(x)
        x = LeakyReLU(0.2)(x)

        # architecture of tail stem
        out = Dense(units=1, kernel_initializer=VarianceScaling(scale=1))(x)
        y = Dense(units=1, kernel_initializer=VarianceScaling(scale=1))(class_in)

        target_dim = x.shape[-1]
        y = Lambda(lambda x: K.tile(x, (1, target_dim)))(y)
        yh = Multiply()([y, x])
        yh = Lambda(lambda x: K.sum(x, axis=1), output_shape=(1, ))(yh)
        model_out = Add()([out, yh])

        self.discriminator = Model(
            [full_resolution_in, onetwentyeightin, sixtyfourin, thirtytwoin, sixteenin, eightin, fourin, class_in],
            model_out
            )
        self.frozen_discriminator = Network(
            [full_resolution_in, onetwentyeightin, sixtyfourin, thirtytwoin, sixteenin, eightin, fourin, class_in],
            model_out
            )
        print(self.discriminator.summary())

    def build_model(self):
        d_optimizer = Adam(lr=self.d_lr, beta_1=0.0, beta_2=0.99)
        g_optimizer = Adam(lr=self.g_lr, beta_1=0.0, beta_2=0.99)

        # build complete discriminator
        full_resolution_real = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        onetwentyeight_real = Input(shape=(self.img_dim_x//2, self.img_dim_y//2, self.img_depth))
        sixtyfour_real = Input(shape=(self.img_dim_x//4, self.img_dim_y//4, self.img_depth))
        thirtytwo_real = Input(shape=(self.img_dim_x//8, self.img_dim_y//8, self.img_depth))
        sixteen_real = Input(shape=(self.img_dim_x//16, self.img_dim_y//16, self.img_depth))
        eight_real = Input(shape=(self.img_dim_x//32, self.img_dim_y//32, self.img_depth))
        four_real = Input(shape=(self.img_dim_x//64, self.img_dim_y//64, self.img_depth))

        full_resolution_fake = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        onetwentyeight_fake = Input(shape=(self.img_dim_x//2, self.img_dim_y//2, self.img_depth))
        sixtyfour_fake = Input(shape=(self.img_dim_x//4, self.img_dim_y//4, self.img_depth))
        thirtytwo_fake = Input(shape=(self.img_dim_x//8, self.img_dim_y//8, self.img_depth))
        sixteen_fake = Input(shape=(self.img_dim_x//16, self.img_dim_y//16, self.img_depth))
        eight_fake = Input(shape=(self.img_dim_x//32, self.img_dim_y//32, self.img_depth))
        four_fake = Input(shape=(self.img_dim_x//64, self.img_dim_y//64, self.img_depth))

        class_in = Input(shape=(self.n_classes, ))
        real_label = self.discriminator(
            [full_resolution_real, onetwentyeight_real, sixtyfour_real, thirtytwo_real, sixteen_real, eight_real, four_real, class_in]
            )
        fake_label = self.discriminator(
            [full_resolution_fake, onetwentyeight_fake, sixtyfour_fake, thirtytwo_fake, sixteen_fake, eight_fake, four_fake, class_in]
            )

        self.discriminator_model = Model(
            [full_resolution_real, onetwentyeight_real, sixtyfour_real, thirtytwo_real, sixteen_real, eight_real, four_real, \
             full_resolution_fake, onetwentyeight_fake, sixtyfour_fake, thirtytwo_fake, sixteen_fake, eight_fake, four_fake, \
             class_in],
            [real_label, fake_label, real_label]
            )
        self.discriminator_model.compile(
            d_optimizer,
            loss=[
                hinge_real_discriminator_loss,
                hinge_fake_discriminator_loss,
                partial(gradient_penalty_loss, averaged_samples=full_resolution_real)
                ],
            loss_weights=[1, 1, self.gp_weight]
            )

        self.frozen_discriminator.trainable = False

        # build generator model
        style_in = Input(shape=(self.latent_dim, ))
        class_in = Input(shape=(self.n_classes, ))
        fake_imgs = self.generator([style_in, class_in])
        fake_imgs.append(class_in)
        frozen_fake_label = self.frozen_discriminator(fake_imgs)

        self.generator_model = Model([style_in, class_in], frozen_fake_label)
        self.generator_model.compile(g_optimizer, hinge_generator_loss)

        print(self.generator_model.summary())
        print(self.discriminator_model.summary())

    ###############################
    ## All our training, etc
    ###############################       

    def train(self, epochs):

        card_generator = CardGenerator(
            img_dir=self.training_dir,
            batch_size=self.batch_size,
            n_cpu=self.n_cpu,
            img_dim=self.img_dim_x,
            multiscale=True
            )
        img_generator = ImgGenerator(
            img_dir='agglomerated_images',
            batch_size=self.batch_size,
            n_cpu=self.n_cpu,
            img_dim=self.img_dim_x,
            multiscale=True
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

                style = np.random.normal(0, 1, size=(self.batch_size, self.latent_dim))
                dummy = np.zeros(shape=(self.batch_size, ))
                fake_batch = self.generator.predict([style, real_labels])
                d_batch = real_batch + fake_batch + [real_labels]
                d_loss = self.discriminator_model.train_on_batch(
                    d_batch,
                    [dummy, dummy, dummy]
                    )
                d_loss_accum.append(d_loss[0])
            
                g_loss = self.generator_model.train_on_batch([style, real_labels], dummy)
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

        card_generator.end()                

    def noise_validation(self, epoch):
        print('Generating Images...')
        if not os.path.isdir(self.validation_dir):
            os.mkdir(self.validation_dir)
        predicted_imgs, _, _, _, _, _, _ = self.generator.predict([self.style_samples, self.label_samples])
        predicted_imgs = [((img+1)*127.5).astype(np.uint8) for img in predicted_imgs]

        # fill a grid
        grid_dim = int(np.sqrt(self.n_noise_samples))
        img_grid = np.zeros(shape=(
            self.img_dim_x*grid_dim, 
            self.img_dim_y*grid_dim,
            self.img_depth
            ))

        positions = itertools.product(range(grid_dim), range(grid_dim))
        for (x_i, y_i), img in zip(positions, predicted_imgs):
            x = x_i * self.img_dim_x
            y = y_i * self.img_dim_y
            img_grid[y:y+self.img_dim_y, x:x+self.img_dim_x, :] = img

        img_grid = Image.fromarray(img_grid.astype(np.uint8))
        img_grid.save(os.path.join(self.validation_dir, "{}_validation_img_{}.png".format(self.model_name, epoch)))

    def save_model_weights(self, epoch, d_loss, g_loss):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        generator_savename = os.path.join(
            self.checkpoint_dir,
            '{}_generator_weights_{}_{:.3f}.h5'.format(self.model_name, epoch, g_loss)
            )
        discriminator_savename = os.path.join(
            self.checkpoint_dir,
            '{}_discriminator_weights_{}_{:.3f}.h5'.format(self.model_name, epoch, d_loss)
            )
        self.generator.save_weights(generator_savename)
        self.discriminator.save_weights(discriminator_savename)
