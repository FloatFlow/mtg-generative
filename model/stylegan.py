from model.utils import *
from model.layers import *
from model.network_blocks import *
from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer
from keras.layers import Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.layers import Concatenate, Embedding, Flatten, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal, VarianceScaling
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
        
        
        self.style_samples = np.random.normal(0, 0.8, size=(self.n_noise_samples, self.z_len))
        self.label_samples = label_generator(self.n_noise_samples, seed=42)
        self.model_name = 'stylegan'
        self.latent_type = 'constant' # or learned
        self.gp_weight = 5
        self.loss_type = 'nonsaturating'
        self.kernel_init = VarianceScaling(scale=np.sqrt(2), mode='fan_in', distribution='normal')


    ###############################
    ## All our architecture
    ###############################

    '''
    To get # of filters to match:
    G:      D:
    1024    64
    2048    128
    '''
    def build_generator(self, ch=128):
        style_in = Input(shape=(self.z_len, ))
        label_in = Input(shape=(self.n_classes, ))
        label_embed = Dense(self.n_classes, kernel_initializer=self.kernel_init)(label_in)
        style = Concatenate(axis=-1)([style_in, label_embed])
        style = LatentPixelNormalization()(style)
        style = Dense(self.z_len, kernel_initializer=self.kernel_init)(style)
        style = LeakyReLU(0.2)(style)
        style = Dense(self.z_len, kernel_initializer=self.kernel_init)(style)
        style = LeakyReLU(0.2)(style)
        style = Dense(self.z_len, kernel_initializer=self.kernel_init)(style)
        style = LeakyReLU(0.2)(style)
        style = Dense(self.z_len, kernel_initializer=self.kernel_init)(style)
        style = LeakyReLU(0.2)(style)


        latent_in = Input(shape=(self.z_len, ))
        if self.latent_type == 'learned':
            x = LearnedConstantLatent()(latent_in)
            x = Dense(units=4*4*ch, kernel_initializer=VarianceScaling(scale=np.sqrt(2)/4, mode='fan_in', distribution='normal'))(x)
            x = Reshape((4, 4, -1))(x)
        else:
            x = ConstantLatent()(latent_in)
            x = epilogue_block(x, style)

        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init, upsample=False) #4x128
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init) #8x128
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init) #16x128
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init) #32x128
        ch = ch // 2
        x = style_generator_block(x, style, ch, kernel_init=self.kernel_init) #64x64
        ch = ch // 2
        x = style_generator_block(x, style, ch,  kernel_init=self.kernel_init) #128x32
        ch = ch // 2
        x = style_generator_block(x, style, ch,  kernel_init=self.kernel_init) #256x16

        x = Conv2D(filters=3,
                   kernel_size=1,
                   padding='same',
                   kernel_initializer=VarianceScaling(scale=1, mode='fan_in', distribution='normal'))(x)
        model_out = Activation('tanh')(x)

        self.generator = Model([style_in, label_in, latent_in], model_out)   
        print(self.generator.summary())
        with open('{}_architecture.txt'.format(self.name), 'w') as f:
            self.generator.summary(print_fn=lambda x: f.write(x + '\n'))


    def build_discriminator(self, ch=16, kernel_init='he_normal'):
        img_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        class_in = Input(shape=(self.n_classes, ))
        
        x = Conv2D(filters=ch,
                   kernel_size=1,
                   padding='same',
                   kernel_initializer=self.kernel_init)(img_in)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(filters=ch,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer=self.kernel_init)(x)
        x = LeakyReLU(0.2)(x)

        ch *= 2
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init) #128x32
        ch *= 2
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init) #64x64
        ch *= 2
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init) #32x128
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init) #16x128
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init) #8x128
        x = style_discriminator_block(x, ch, kernel_init=self.kernel_init) #4x128
        x = MiniBatchStd()(x)
        x = Conv2D(filters=ch,
                   kernel_size=3,
                   padding='valid',
                   kernel_initializer=self.kernel_init)(x)
        x = LeakyReLU(0.2)(x)
        #x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(units=ch, kernel_initializer=self.kernel_init)(x)
        x = LeakyReLU(0.2)(x)

        # architecture of tail stem
        out = Dense(units=1, kernel_initializer=VarianceScaling(scale=1, mode='fan_in', distribution='normal'))(x)
        y = Dense(units=1, kernel_initializer=VarianceScaling(scale=1, mode='fan_in', distribution='normal'))(class_in)

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
        if self.loss_type == 'hinge':
            loss_collection = [hinge_real_discriminator_loss,
                               hinge_fake_discriminator_loss,
                               hinge_generator_loss]
        elif self.loss_type == 'wgan':
            loss_collection = [wgan_loss, wgan_loss, wgan_loss]
        elif self.loss_type == 'nonsaturating':
            loss_collection = [nonsat_real_discriminator_loss,
                               nonsat_fake_discriminator_loss,
                               nonsat_generator_loss]
        elif self.loss_type == 'mse':
            loss_collection = ['mse', 'mse', 'mse']
        else:
            raise ValueError('{} is an invalid loss type. \
                             Choose between "hinge","wgan", or "nonsaturating".'.format(self.loss_type))

        # build complete discriminator
        fake_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        real_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        class_in = Input(shape=(self.n_classes,))
        fake_label = self.discriminator([fake_in, class_in])
        real_label = self.discriminator([real_in, class_in])

        self.discriminator_model = Model([real_in, fake_in, class_in], [real_label, fake_label, real_label])
        self.discriminator_model.compile(d_optimizer,
                                         loss=[loss_collection[0],
                                               loss_collection[1],
                                               partial(gradient_penalty_loss, averaged_samples=real_in)],
                                         loss_weights=[1, 1, self.gp_weight])

        self.frozen_discriminator.trainable = False

        # build generator model
        style_in = Input(shape=(self.z_len, ))
        latent_in = Input(shape=(self.z_len, ))
        class_in = Input(shape=(self.n_classes, ))
        fake_img = self.generator([style_in, class_in, latent_in])
        frozen_fake_label = self.frozen_discriminator([fake_img, class_in])

        self.generator_model = Model([style_in, latent_in, class_in], frozen_fake_label)
        self.generator_model.compile(g_optimizer, loss_collection[2])
        
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
                dummy_latent = np.ones(shape=(self.batch_size, self.z_len))
                dummy = np.ones(shape=(self.batch_size, ))
                ones = np.ones(shape=(self.batch_size, ))
                zeros = np.zeros(shape=(self.batch_size, ))
                neg_ones = -ones

                fake_batch = self.generator.predict([style, real_labels, dummy_latent])
                
                d_loss = self.discriminator_model.train_on_batch([real_batch, fake_batch, real_labels],
                                                                 [ones, neg_ones, dummy])
                d_loss = sum(d_loss)
                d_loss_accum.append(d_loss)
            
                g_loss = self.generator_model.train_on_batch([style, dummy_latent, real_labels], ones)
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
        dummy_latent = np.ones(shape=(self.noise_samples.shape[0], self.z_len))
        predicted_imgs = self.generator.predict([self.style_samples, self.label_samples, dummy_latent])
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
        img_grid.save(os.path.join(self.validation_dir, "{}_validation_img_{}.png".format(self.model_name, epoch)))


    def noise_validation_wlabel(self):
        pass

    def predict_noise_wlabel_testing(self, label):
        pass

    def save_model_weights(self, epoch, d_loss, g_loss):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        generator_savename = os.path.join(self.checkpoint_dir,
                                          '{}_{}_generator_weights_{}_{:.3f}.h5'.format(self.model_name,
                                                                                        self.loss_type,
                                                                                        epoch,
                                                                                        g_loss))
        discriminator_savename = os.path.join(self.checkpoint_dir,
                                          '{}_{}_discriminator_weights_{}_{:.3f}.h5'.format(self.model_name,
                                                                                        self.loss_type,
                                                                                        epoch,
                                                                                        d_loss))
        self.generator.save_weights(generator_savename)
        self.discriminator.save_weights(discriminator_savename)

    def load_model_weights(self, g_weight_path, d_weight_path):
        self.generator.load_weights(g_weight_path, by_name=True)
        self.discriminator.load_weights(d_weight_path, by_name=True)
