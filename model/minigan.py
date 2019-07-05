from model.utils import *
from model.layers import *
import argparse
from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer
from keras.layers import Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.layers import Concatenate, Embedding, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.constraints import MinMaxNorm
from keras.initializers import RandomNormal
import keras.backend as K
from keras.engine.network import Network
from tqdm import tqdm
import cv2
from PIL import Image
from keras.utils import plot_model



class miniGAN():
    def __init__(self, 
                 img_dim_x,
                 img_dim_y,
                 img_depth,
                 z_len,
                 n_classes,
                 n_noise_samples,
                 resblock_up_squeeze,
                 resblock_down_squeeze,
                 g_lr,
                 d_lr,
                 batch_size,
                 g_d_update_ratio,
                 d_g_update_ratio,
                 save_freq,
                 training_dir,
                 validation_dir,
                 checkpoint_dir,
                 testing_dir,
                 conditional,
                 normalization,
                 kernel_init,
                 feat_matching,
                 **kwargs):
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.img_depth = img_depth
        self.z_len = z_len
        self.n_noise_samples = n_noise_samples
        self.resblock_up_squeeze = resblock_up_squeeze
        self.resblock_down_squeeze = resblock_down_squeeze
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.batch_size = batch_size
        self.d_g_update_ratio = d_g_update_ratio
        self.g_d_update_ratio = g_d_update_ratio
        self.save_freq = save_freq
        self.training_dir = training_dir
        self.validation_dir = validation_dir
        self.checkpoint_dir = checkpoint_dir
        self.testing_dir = testing_dir
        self.kernel_reg = None
        self.conditional = conditional
        self.n_classes = n_classes
        self.normalization = normalization
        self.feat_matching = feat_matching
        if (self.conditional == 'biggan') or (self.conditional == 'onehot') :
            self.noise_samples = np.concatenate([np.random.normal(0,0.8,size=(self.n_noise_samples, self.z_len)), \
                                                label_generator(self.n_noise_samples, seed=42)], axis=1)
        else:
            self.noise_samples = np.random.normal(0,1,size=(self.n_noise_samples, self.z_len))

        if kernel_init == 'norm':
            self.kernel_init = RandomNormal(mean=0.0, stddev=0.02, seed=None)
        elif kernel_init == 'ortho':
            self.kernel_init = 'orthogonal'
        else:
            print('Kernel Initializer Argument Not Recognized!')


    ###############################
    ## All our architecture
    ###############################
    def batchnorm_zproj(self, x, z):
        # for some reason keras batchnorm fails when setting scaling or bias to none/false
        if self.normalization == 'batchnorm':
            x = BatchNormalization(gamma_constraint=MinMaxNorm(min_value=1.0, max_value=1.0, rate=1.0, axis=0),
                                   beta_constraint=MinMaxNorm(min_value=0.0, max_value=0.0, rate=1.0, axis=0))(x)
        if self.normalization == 'pixelnorm':
            x = PixelNormalization()(x)
        if self.normalization == 'instancenorm':
            x = InstanceNormalization(scale=False, center=False)(x)

        # note bigGAN init gamma and beta kernels as N(0, 0.02)
        target_shape = K.int_shape(x)
        tile_shape = (1, target_shape[1], target_shape[2], 1)
        gamma = Dense(target_shape[-1], use_bias=True, kernel_initializer='ones')(z)
        gamma = Reshape((1, 1, -1))(gamma)
        beta = Dense(target_shape[-1], use_bias=True, kernel_initializer='zeros')(z)
        beta = Reshape((1, 1, -1))(beta)

        if self.normalization in ['batchnorm', 'pixelnorm', 'instancenorm']:
            gamma = Lambda(lambda x: K.tile(x, tile_shape))(gamma)
            beta = Lambda(lambda x: K.tile(x, tile_shape))(beta)
            x = Multiply()([x, gamma]) 
            x = Add()([x, beta])
        else:
            x = AdaInstanceNormalization()([x, beta, gamma])
        return x



    def deep_resblock_up(self, x, z, ch, c=None, upsample=True):
        # left path
        xl = Lambda(lambda x: x[:,:,:,:ch])(x)
        if upsample:
            xl = UpSampling2D((2,2), interpolation='nearest')(xl)

        # right path
        
        xr = self.batchnorm_zproj(x, z)
        xr = Activation('relu')(x)
        xr = ConvSN2D(filters=ch//self.resblock_up_squeeze,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=self.kernel_init,
                      kernel_regularizer=self.kernel_reg)(xr)

        xr = self.batchnorm_zproj(xr, z)
        xr = Activation('relu')(xr)
        if upsample:
            xr = UpSampling2D((2,2), interpolation='nearest')(xr)
        xr = ConvSN2D(filters=ch//self.resblock_up_squeeze,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=self.kernel_init,
                      kernel_regularizer=self.kernel_reg)(xr)

        xr = self.batchnorm_zproj(xr, z)
        xr = Activation('relu')(xr)
        xr = ConvSN2D(filters=ch//self.resblock_up_squeeze,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=self.kernel_init,
                      kernel_regularizer=self.kernel_reg)(xr)

        xr = self.batchnorm_zproj(xr, z)
        xr = Activation('relu')(xr)
        xr = ConvSN2D(filters=ch,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=self.kernel_init,
                      kernel_regularizer=self.kernel_reg)(xr)

        x = Add()([xl, xr])
        return x


    def deep_resblock_down(self, x, ch, downsample=True):
        # left path
        if downsample:
            xl = AveragePooling2D((2,2))(x)
        else:
            xl = x
        input_channels = K.int_shape(xl)[-1]
        add_channels = ch-input_channels
        if add_channels > 0:
            xl_l = ConvSN2D(filters=add_channels,
                            kernel_size=1,
                            strides=1,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=self.kernel_init)(xl)
            xl = Concatenate()([xl, xl_l])

        # right path
        xr = Activation('relu')(x)
        xr = ConvSN2D(filters=ch//self.resblock_down_squeeze,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=self.kernel_init)(xr)

        xr = Activation('relu')(xr)
        xr = ConvSN2D(filters=ch//self.resblock_down_squeeze,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=self.kernel_init)(xr)

        xr = Activation('relu')(xr)
        xr = ConvSN2D(filters=ch//self.resblock_down_squeeze,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=self.kernel_init)(xr)

        xr = Activation('relu')(xr)
        if downsample:
            xr = AveragePooling2D((2,2))(xr)
        xr = ConvSN2D(filters=ch,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=self.kernel_init)(xr)

        x = Add()([xl, xr])
        return x

    '''
    To get # of filters to match:
    G:      D:
    1024    64
    2048    128
    '''
    def build_deep_generator(self, ch=1024):
        if self.conditional in ('biggan', 'onehot'):
            model_in = Input(shape=(self.z_len+self.n_classes,))
        else:
            model_in = Input(shape=(self.z_len,))
        x = Dense(4*4*ch)(model_in)
        x = Reshape((4,4,-1))(x)

        #x = self.deep_resblock_up(x, model_in, ch, upsample=False)
        x = self.deep_resblock_up(x, model_in, ch, upsample=True)

        #x = self.deep_resblock_up(x, model_in, ch, upsample=False)
        ch = ch//2
        x = self.deep_resblock_up(x, model_in, ch, upsample=True)

        #x = self.deep_resblock_up(x, model_in, ch, upsample=False)
        #ch = ch//2 # no downchannel in biggan
        x = self.deep_resblock_up(x, model_in, ch, upsample=True)

        #x = self.deep_resblock_up(x, model_in, ch, upsample=False)
        ch = ch//2
        x = self.deep_resblock_up(x, model_in, ch, upsample=True)

        x = Attention_2SN(ch)(x)

        #x = self.deep_resblock_up(x, model_in, ch, upsample=False)
        ch = ch//2
        x = self.deep_resblock_up(x, model_in, ch, upsample=True)

        #x = self.deep_resblock_up(x, model_in, ch, upsample=False)
        ch = ch//2
        x = self.deep_resblock_up(x, model_in, ch, upsample=True)

        if self.normalization == 'batchnorm':
            x = BatchNormalization()(x)
        if self.normalization == 'pixelnorm':
            x = PixelNormalization()(x)
        x = Activation('relu')(x)
        model_out = ConvSN2D(filters=3,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     use_bias=False,
                     kernel_initializer=self.kernel_init,
                     kernel_regularizer=self.kernel_reg,
                     activation='tanh')(x)

        self.generator = Model(model_in, model_out)   
        print(self.generator.summary())   


    def build_deep_discriminator(self, ch=64):
        model_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        if self.conditional == 'biggan':
            class_in = Input(shape=(self.n_classes,))
        x = ConvSN2D(filters=ch,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     use_bias=False,
                     kernel_initializer=self.kernel_init)(model_in)
        ch *= 2
        x = self.deep_resblock_down(x, ch, downsample=True)
        #x = self.deep_resblock_down(x, ch, downsample=False)

        ch *= 2
        x = self.deep_resblock_down(x, ch, downsample=True)
        #x = self.deep_resblock_down(x, ch, downsample=False)

        x = Attention_2SN(ch)(x)

        ch *= 2
        x = self.deep_resblock_down(x, ch, downsample=True)
        #x = self.deep_resblock_down(x, ch, downsample=False)

        #ch *= 2 # no channel scaling in biggan
        x = self.deep_resblock_down(x, ch, downsample=True)
        #x = self.deep_resblock_down(x, ch, downsample=False)

        ch *= 2
        x = self.deep_resblock_down(x, ch, downsample=True)
        #x = self.deep_resblock_down(x, ch, downsample=False)

        x = self.deep_resblock_down(x, ch, downsample=True)
        #x = self.deep_resblock_down(x, ch, downsample=False)

        x = Activation('relu')(x)
        x = GlobalSumPooling2D()(x)
        x = MinibatchDiscrimination(16, 16)(x)

        # architecture of tail stem
        if self.conditional == 'biggan':
            out = Dense(1)(x)
            #print('Pooling shape: {}'.format(x.shape))
            y = Dense(1, use_bias=False)(class_in)
            #print('Embedding shape: {}'.format(y.shape))
            target_dim = x.shape[-1]
            y = Lambda(lambda x: K.tile(x, (1, target_dim)))(y)
            yh = Multiply()([y, x])
            #yh = Lambda(lambda x: K.sum(x, axis=[1]), output_shape=(self.batch_size,))(yh)
            yh = Lambda(lambda x: K.sum(x, axis=1), output_shape=(1,))(yh)
            model_out = Add()([out, yh])

            self.discriminator = Model([model_in, class_in], model_out)
            self.frozen_discriminator = Network([model_in, class_in], model_out)
        elif self.conditional == 'onehot':
            model_out = Dense(1)(x)
            classifier = Dense(self.n_classes, activation='sigmoid')(x)

            self.discriminator = Model(model_in, [model_out, classifier])
            self.frozen_discriminator = Network(model_in, [model_out, classifier])
        else:
            model_out = Dense(1)(x)
            self.discriminator = Model(model_in, model_out)
            self.frozen_discriminator = Network(model_in, model_out)

        # whether to use feature matching
        if self.feat_matching == 'True':
            self.feature_matcher = Model(model_in, x)
        print(self.discriminator.summary())

    def build_model(self):
        d_optimizer = Adam(lr=self.d_lr, beta_1=0.0, beta_2=0.9)
        g_optimizer = Adam(lr=self.g_lr, beta_1=0.0, beta_2=0.9)

        # build complete discriminator
        if self.conditional == 'biggan':
            fake_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
            real_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
            class_in = Input(shape=(self.n_classes,))
            fake_label = self.discriminator([fake_in, class_in])
            real_label = self.discriminator([real_in, class_in])

            self.discriminator_model = Model([real_in, fake_in, class_in], [real_label, fake_label])
            self.discriminator_model.compile(d_optimizer, loss=[real_discriminator_loss, fake_discriminator_loss])
        elif self.conditional == 'onehot':
            fake_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
            real_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
            fake_label, fake_classes = self.discriminator(fake_in)
            real_label, real_classes = self.discriminator(real_in)

            self.discriminator_model = Model([real_in, fake_in], [real_label, fake_label, real_classes])
            self.discriminator_model.compile(d_optimizer,
                                            loss=[real_discriminator_loss, fake_discriminator_loss, 'categorical_crossentropy'],
                                            loss_weights=[1,1,1e-3])
        else:
            fake_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
            real_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
            fake_label = self.discriminator(fake_in)
            real_label = self.discriminator(real_in)

            self.discriminator_model = Model([real_in, fake_in], [real_label, fake_label])
            self.discriminator_model.compile(d_optimizer, loss=[real_discriminator_loss, fake_discriminator_loss])

        # build generator model
        self.frozen_discriminator.trainable = False

        if self.conditional == 'biggan':
            z_in = Input(shape=(self.z_len+self.n_classes,))
            class_in = Input(shape=(self.n_classes,))
            fake_img = self.generator(z_in)
            frozen_fake_label = self.frozen_discriminator([fake_img,class_in])
            if self.feat_matching == 'True':
                frozen_fake_features = self.feature_matcher(fake_img)
                self.generator_model = Model([z_in, class_in], [frozen_fake_label, frozen_fake_features])
                self.generator_model.compile(g_optimizer, 
                                             [generator_loss, 'mse'],
                                             loss_weights=[1,1e-3])
            else:
                self.generator_model = Model([z_in, class_in], frozen_fake_label)
                self.generator_model.compile(g_optimizer, generator_loss)
        elif self.conditional == 'onehot':
            z_in = Input(shape=(self.z_len+self.n_classes,))
            fake_img = self.generator(z_in)
            frozen_fake_label, frozen_fake_class = self.frozen_discriminator(fake_img)
            if self.feat_matching == 'True':
                frozen_fake_features = self.feature_matcher(fake_img)
                self.generator_model = Model(z_in, [frozen_fake_label, frozen_fake_class, frozen_fake_features])
                self.generator_model.compile(g_optimizer, 
                                             [generator_loss,'categorical_crossentropy','mse'],
                                             loss_weights=[1,1e-3,1e-3])
            else:
                self.generator_model = Model(z_in, [frozen_fake_label, frozen_fake_class])
                self.generator_model.compile(g_optimizer, 
                                             loss=[generator_loss, 'categorical_crossentropy'],
                                             loss_weights=[1,1e-3])
        else:
            z_in = Input(shape=(self.z_len,))
            fake_img = self.generator(z_in)
            frozen_fake_label = self.frozen_discriminator(fake_img)
            self.generator_model = Model(z_in, frozen_fake_label)
            self.generator_model.compile(g_optimizer, generator_loss)
        print(self.discriminator_model.summary())
        print(self.generator_model.summary())
        '''
        plot_model(self.discriminator_model, 
                   to_file='minigan_discriminator_model.png')
        plot_model(self.generator_model, 
                   to_file='minigan_generator_model.png')
        plot_model(self.generator, 
                   to_file='minigan_generator.png')
        plot_model(self.discriminator, 
                   to_file='minigan_discriminator.png')
        '''

    ###############################
    ## All our training, etc
    ###############################
    def train(self, epochs):

        card_generator = CardGenerator(img_dir = self.training_dir,
                                       batch_size = self.batch_size, 
                                       n_threads = 4,
                                       n_cpu = 4, 
                                       img_dim = self.img_dim_x)
        n_batches = card_generator.n_batches
        for epoch in range(epochs):
            d_loss_accum = []
            g_loss_accum = []

            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                real_batch, real_labels = card_generator.next()

                noise = np.random.normal(0,1,size=(self.batch_size, self.z_len))
                noise_labels = np.concatenate([noise, real_labels], axis=1)
                dummy = np.ones(shape=(self.batch_size,))

                fake_batch = self.generator.predict(noise_labels)
                
                d_loss = self.discriminator_model.train_on_batch([real_batch, fake_batch, real_labels], [dummy, dummy])
                d_loss = sum(d_loss)
                d_loss_accum.append(d_loss)
            
                g_loss = self.generator_model.train_on_batch([noise_labels, real_labels], dummy)
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

            card_generator.shuffle(epoch)

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
