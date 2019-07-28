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


class StyleGAN():
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
        self.name = 'stylegan'
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
        
        self.noise_samples = np.concatenate([np.random.normal(0, 0.8, size=(self.n_noise_samples, self.z_len)), \
                                            label_generator(self.n_noise_samples, seed=42)], axis=1)

        self.kernel_init = 'he_normal'


    ###############################
    ## All our architecture
    ###############################

    '''
    To get # of filters to match:
    G:      D:
    1024    64
    2048    128
    '''
    def build_generator(self, ch=128, kernel_init='he_normal'):
        style_in = Input(shape=(self.z_len+self.n_classes, ))
        style = Dense(self.z_len)(style_in)
        style = LeakyReLU(0.2)(style)
        style = Dense(self.z_len)(style)
        style = LeakyReLU(0.2)(style)

        noise_in = Input(shape=(self.img_dim_x, self.img_dim_y, 1))

        latent_in = Input(shape=(self.z_len, ))
        x = LearnedConstantLatent()(latent_in)
        x = Dense(4*4*ch)(x)
        x = Reshape((4, 4, -1))(x)

        x = Conv2D(filters=ch,
                   kernel_size=4,
                   padding='same',
                   kernel_initializer=kernel_init)(x)
        x = LeakyReLU(0.2)(x)
        x = style_generator_block(x, style, noise_in, ch) #8x128
        x = style_generator_block(x, style, noise_in, ch) #16x128
        x = style_generator_block(x, style, noise_in, ch) #32x128
        ch = ch // 2
        x = style_generator_block(x, style, noise_in, ch) #64x64
        ch = ch // 2
        x = style_generator_block(x, style, noise_in, ch) #128x32
        ch = ch // 2
        x = style_generator_block(x, style, noise_in, ch) #256x16

        x = Conv2D(filters=3,
                   kernel_size=1,
                   padding='same',
                   kernel_initializer=kernel_init)(x)
        model_out = Activation('tanh')(x)

        self.generator = Model([style_in, latent_in, noise_in], model_out)   
        print(self.generator.summary())
        with open('{}_architecture.txt'.format(self.name), 'w') as f:
            self.generator.summary(print_fn=lambda x: f.write(x + '\n'))


    def build_discriminator(self, ch=16, kernel_init='he_normal'):
        img_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        class_in = Input(shape=(self.n_classes, ))
        
        x = Conv2D(filters=ch,
                   kernel_size=1,
                   padding='same',
                   kernel_initializer=kernel_init)(img_in)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(filters=ch,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer=kernel_init)(x)
        x = LeakyReLU(0.2)(x)

        ch *= 2
        x = style_discriminator_block(x, ch) #128x32
        ch *= 2
        x = style_discriminator_block(x, ch) #64x64
        ch *= 2
        x = style_discriminator_block(x, ch) #32x128
        x = style_discriminator_block(x, ch) #16x128
        x = style_discriminator_block(x, ch) #8x128
        x = style_discriminator_block(x, ch) #4x128
        x = MiniBatchStd()(x)
        x = Conv2D(filters=ch,
                   kernel_size=4,
                   padding='valid',
                   kernel_initializer=kernel_init)(x)
        x = LeakyReLU(0.2)(x)
        x = GlobalAveragePooling2D()(x)

        # architecture of tail stem
        out = Dense(1)(x)
        y = Dense(1)(class_in)

        target_dim = x.shape[-1]
        y = Lambda(lambda x: K.tile(x, (1, target_dim)))(y)
        yh = Multiply()([y, x])
        yh = Lambda(lambda x: K.sum(x, axis=1), output_shape=(1, ))(yh)
        model_out = Add()([out, yh])

        self.discriminator = Model([img_in, class_in], model_out)
        self.frozen_discriminator = Network([img_in, class_in], model_out)

        print(self.discriminator.summary())
        with open('{}_architecture.txt'.format(self.name), 'a') as f:
            self.discriminator.summary(print_fn=lambda x: f.write(x + '\n'))

    def build_model(self):
        d_optimizer = Adam(lr=self.d_lr, beta_1=0.0, beta_2=0.9)
        g_optimizer = Adam(lr=self.g_lr, beta_1=0.0, beta_2=0.9)

        # build complete discriminator
        fake_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        real_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        class_in = Input(shape=(self.n_classes,))
        fake_label = self.discriminator([fake_in, class_in])
        real_label = self.discriminator([real_in, class_in])

        self.discriminator_model = Model([real_in, fake_in, class_in], [real_label, fake_label, real_label])
        self.discriminator_model.compile(d_optimizer,
                                         loss=[hinge_real_discriminator_loss,
                                               hinge_fake_discriminator_loss,
                                               partial(gradient_penalty_loss, averaged_samples=real_in)],
                                         loss_weights=[1, 1, 10])

        self.frozen_discriminator.trainable = False

        # build generator model
        style_in = Input(shape=(self.z_len+self.n_classes,))
        latent_in = Input(shape=(self.z_len, ))
        noise_in = Input(shape=(self.img_dim_x, self.img_dim_y, 1))
        class_in = Input(shape=(self.n_classes,))
        fake_img = self.generator([style_in, latent_in, noise_in])
        frozen_fake_label = self.frozen_discriminator([fake_img, class_in])

        self.generator_model = Model([style_in, latent_in, noise_in, class_in], frozen_fake_label)
        self.generator_model.compile(g_optimizer, hinge_generator_loss)
        
        print(self.discriminator_model.summary())
        print(self.generator_model.summary())

    ###############################
    ## All our training, etc
    ###############################       

    def train(self, epochs):

        card_generator = CardGenerator(img_dir=self.training_dir,
                                       batch_size=self.batch_size,
                                       n_cpu=self.n_cpu,
                                       img_dim=self.img_dim_x)
        n_batches = card_generator.n_batches
        for epoch in range(epochs):
            d_loss_accum = []
            g_loss_accum = []

            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                real_batch, real_labels = card_generator.next()

                style = np.random.normal(0, 1, size=(self.batch_size, self.z_len))
                style_labels = np.concatenate([style, real_labels], axis=1)
                noise = np.random.normal(0, 1, size=(self.batch_size, self.img_dim_x, self.img_dim_y, 1))
                dummy_latent = np.ones(shape=(self.batch_size, self.z_len))
                dummy = np.ones(shape=(self.batch_size, ))
                ones = np.ones(shape=(self.batch_size, ))
                zeros = np.zeros(shape=(self.batch_size, ))
                neg_ones = -ones

                fake_batch = self.generator.predict([style_labels, dummy_latent, noise])
                
                d_loss = self.discriminator_model.train_on_batch([real_batch, fake_batch, real_labels],
                                                                 [ones, neg_ones, dummy])
                d_loss = sum(d_loss)
                d_loss_accum.append(d_loss)
            
                g_loss = self.generator_model.train_on_batch([style_labels, dummy_latent, noise, real_labels], zeros)
                g_loss_accum.append(g_loss)
                

                pbar.update()
            pbar.close()

            print('{}/{} ----> d_loss: {}, g_loss: {}'.format(epoch, 
                                                              epochs, 
                                                              np.mean(d_loss_accum), 
                                                              np.mean(g_loss_accum)))

            if epoch % self.save_freq == 0:
                self.noise_validation(epoch)
                self.save_model_weights(epoch, d_loss, g_loss)

            card_generator.shuffle()

        card_generator.end()                



    def noise_validation(self, epoch):
        print('Generating Images...')
        if not os.path.isdir(self.validation_dir):
            os.mkdir(self.validation_dir)
        latent = np.ones(self.noise_samples.shape[0], self.z_len)
        noise = np.random.normal(0, 1, size=(self.noise_samples.shape[0],
                                             self.img_dim_x,
                                             self.img_dim_y,
                                             1))
        predicted_imgs = self.generator.predict([self.noise_samples, latent, noise])
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
        img_grid.save(os.path.join(self.validation_dir, "validation_img_{}.png".format(epoch)))


    def noise_validation_wlabel(self):
        pass

    def predict_noise_wlabel_testing(self, label):
        pass

    def save_model_weights(self, epoch, d_loss, g_loss):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        generator_savename = os.path.join(self.checkpoint_dir, 'minigan_generator_weights_{}_{:.3f}.h5'.format(epoch, g_loss))
        discriminator_savename = os.path.join(self.checkpoint_dir, 'minigan_discriminator_weights_{}_{:.3f}.h5'.format(epoch, d_loss))
        self.generator.save_weights(generator_savename)
        self.discriminator.save_weights(discriminator_savename)

    def load_model_weights(self, g_weight_path, d_weight_path):
        self.generator.load_weights(g_weight_path, by_name=True)
        self.discriminator.load_weights(d_weight_path, by_name=True)
