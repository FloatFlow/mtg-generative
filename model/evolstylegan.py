"""
A smaller version of biggan
Seems to work better than stylegan on small, diverse datasets
"""
from tqdm import tqdm
import cv2
from PIL import Image
from keras.utils import plot_model
from functools import partial

from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer, Dropout
from keras.layers import Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.layers import Concatenate, Embedding, Flatten, LeakyReLU
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
from keras.engine.network import Network
from keras.applications.inception_resnet_v2 import InceptionResNetV2

#import kuti
#from kuti import applications as apps
import nevergrad as ng

from model.utils import *
from model.layers import *
from model.network_blocks import *

class EvolStyleGAN():
    def __init__(
        self, 
        img_width,
        img_height,
        img_depth,
        lr,
        training_dir,
        validation_dir,
        checkpoint_dir,
        testing_dir,
        n_noise_samples=16
        ):
        self.img_width = img_width
        self.img_height = img_height
        self.img_depth = img_depth
        self.n_noise_samples = n_noise_samples
        self.lr = lr
        self.training_dir = training_dir
        self.validation_dir = validation_dir
        self.checkpoint_dir = checkpoint_dir
        self.testing_dir = testing_dir
        for path in [self.validation_dir,
                     self.checkpoint_dir,
                     self.testing_dir]:
            if not os.path.isdir(path):
                os.makedirs(path)
        self.z_len = 256
        self.n_classes = 5
        self.name = 'evolstylegan'
        self.noise_samples = np.random.normal(0,0.8,size=(self.n_noise_samples, self.z_len))
        self.label_samples = label_generator(self.n_noise_samples)
        self.build_generator()

    ###############################
    ## All our architecture
    ###############################
    def build_generator(self):
        model_in = Input(shape=(self.z_len, ))
        class_in = Input(shape=(self.n_classes, ))
        class_embed = Dense(self.z_len, kernel_initializer='he_uniform')(class_in)
        style = Concatenate()([model_in, class_in])
        style = Lambda(
            lambda x: x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + K.epsilon())
            )(style)
        for _ in range(4):
            style = Dense(units=self.z_len, kernel_initializer='he_uniform')(style)
            style = LeakyReLU(0.2)(style)

        ch = self.z_len
        x = LearnedConstantLatent()(model_in)
        x = style2_generator_layer(x, style, output_dim=ch) #4x256
        to_rgb_4x4 = to_rgb(x, style)
        to_rgb_4x4 = UpSampling2D(2, interpolation='bilinear')(to_rgb_4x4)

        x = style2_generator_layer(x, style, output_dim=ch, upsample=True) #8x256
        x = style2_generator_layer(x, style, output_dim=ch)
        to_rgb_8x8 = to_rgb(x, style)
        to_rgb_8x8 = Add()([to_rgb_8x8, to_rgb_4x4])
        to_rgb_8x8 = UpSampling2D(2, interpolation='bilinear')(to_rgb_8x8)

        x = style2_generator_layer(x, style, output_dim=ch, upsample=True) #16x256
        x = style2_generator_layer(x, style, output_dim=ch)
        to_rgb_16x16 = to_rgb(x, style)
        to_rgb_16x16 = Add()([to_rgb_16x16, to_rgb_8x8])
        to_rgb_16x16 = UpSampling2D(2, interpolation='bilinear')(to_rgb_16x16)

        ch = ch//2
        x = style2_generator_layer(x, style, output_dim=ch, upsample=True) #32x128
        x = style2_generator_layer(x, style, output_dim=ch)
        to_rgb_32x32 = to_rgb(x, style)
        to_rgb_32x32 = Add()([to_rgb_32x32, to_rgb_16x16])
        to_rgb_32x32 = UpSampling2D(2, interpolation='bilinear')(to_rgb_32x32)

        ch = ch//2
        x = style2_generator_layer(x, style, output_dim=ch, upsample=True) #64x64
        x = style2_generator_layer(x, style, output_dim=ch)
        to_rgb_64x64 = to_rgb(x, style)
        to_rgb_64x64 = Add()([to_rgb_64x64, to_rgb_32x32])
        to_rgb_64x64 = UpSampling2D(2, interpolation='bilinear')(to_rgb_64x64)

        ch = ch//2
        x = style2_generator_layer(x, style, output_dim=ch, upsample=True) #128x32
        x = style2_generator_layer(x, style, output_dim=ch)
        to_rgb_128x128 = to_rgb(x, style)
        to_rgb_128x128 = Add()([to_rgb_128x128, to_rgb_64x64])
        to_rgb_128x128 = UpSampling2D(2, interpolation='bilinear')(to_rgb_128x128)

        ch = ch//2
        x = style2_generator_layer(x, style, output_dim=ch, upsample=True) #256x16
        x = style2_generator_layer(x, style, output_dim=ch)
        to_rgb_256x256 = to_rgb(x, style)
        to_rgb_256x256 = Add()([to_rgb_256x256, to_rgb_128x128])
        model_out = Activation('tanh')(to_rgb_256x256)

        self.generator = Model([model_in, class_in], model_out)   
        print(self.generator.summary())

    def load_quality_estimator(self, estimator_path):
        #base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
        #head = apps.fc_layers(base_model.output, name='fc', 
        #                      fc_sizes      = [2048, 1024, 256, 1], 
        #                      dropout_rates = [0.25, 0.25, 0.5, 0], 
        #                      batch_norm    = 2)
        base_model = InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
            )
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(2048, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        qual = Dense(1, activation='linear')(x)

        self.quality_estimator = Model(inputs=base_model.input, outputs=qual)
        self.quality_estimator.load_weights(estimator_path)
        print(self.quality_estimator.summary())

    def optimizer_step(self, z, c):
        g_z = self.generator.predict([z, c])
        g_z = np.stack([cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC) for img in g_z])
        q_z = self.quality_estimator.predict(g_z)
        return -np.squeeze(q_z)

    ###############################
    ## All our training, etc
    ###############################               

    def train(self, epochs, batch_size=4, n_cpu=4, save_freq=5):
        n_samples = 100
        budget = 40
        np.random.seed(42)
        z_0 = np.random.normal(0, 0.5, (n_samples, self.z_len))
        classes = generate_labels(n_samples=n_samples, n_classes=self.n_classes, repeats=2)
        pz_0 = ng.p.Array(init=z_0)
        pz_0.set_bounds(lower=-5, upper=5, method='bouncing')
        pz_0.set_mutation(sigma=1e-6, custom='discrete')
        #print(help(ng.optimizers.OnePlusOne))
        optimizer = ng.optimizers.OnePlusOne(
            parametrization=pz_0,
            budget=budget,
            num_workers=1
            )
        #optimizer = ng.optimization.optimizerlib.ParametrizedOnePlusOne(
        #    parametrization=pz_0,
        #    budget=budget,
        #    noise_handling=('random', 1e-6),
        #    mutation='discrete',
        #    num_workers=n_cpu
        #    )
        recommendations = optimizer.minimize(
            partial(self.optimizer_step, c=classes),
            verbosity=1
            )
        print(recommendations.value.shape)

        self.noise_validation(epoch=0, z=z_0[:16, ...], c=classes[:16, ...])
        self.noise_validation(epoch=1, z=recommendations.value[:16, ...], c=classes[:16, ...])

    def noise_validation(self, epoch, z, c):
        print('Generating Images...')
        if not os.path.isdir(self.validation_dir):
            os.mkdir(self.validation_dir)
        predicted_imgs = self.generator.predict([z, c])
        predicted_imgs = [((img+1)*127.5).astype(np.uint8) for img in predicted_imgs]

        # fill a grid
        grid_dim = int(np.sqrt(self.n_noise_samples))
        img_grid = np.zeros(shape=(self.img_width*grid_dim, 
                                   self.img_height*grid_dim,
                                   self.img_depth))


        positions = itertools.product(range(grid_dim), range(grid_dim))
        for (x_i, y_i), img in zip(positions, predicted_imgs):
            x = x_i * self.img_width
            y = y_i * self.img_height
            img_grid[y:y+self.img_height, x:x+self.img_width, :] = img

        img_grid = Image.fromarray(img_grid.astype(np.uint8))
        img_grid.save(os.path.join(self.validation_dir, "{}_validation_img_{}.png".format(self.name, epoch)))

    def save_model_weights(self, epoch, d_loss, g_loss):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        generator_savename = os.path.join(
            self.checkpoint_dir,
            '{}_generator_weights_{}_{:.3f}.h5'.format(self.name, epoch, g_loss)
            )
        discriminator_savename = os.path.join(
            self.checkpoint_dir,
            '{}_discriminator_weights_{}_{:.3f}.h5'.format(self.name, epoch, d_loss)
            )
        self.generator.save_weights(generator_savename)
        self.discriminator.save_weights(discriminator_savename)
