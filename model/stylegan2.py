"""
A smaller version of biggan
Seems to work better than stylegan on small, diverse datasets
"""
from tqdm import tqdm
import cv2
from PIL import Image
from keras.utils import plot_model
from functools import partial
from scipy import ndimage

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

class StyleGAN2():
    def __init__(
        self, 
        img_width,
        img_height,
        img_depth,
        z_len,
        n_classes,
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
        self.z_len = z_len
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
        self.n_classes = n_classes
        self.name = 'stylegan2'
        self.noise_samples = np.random.normal(0,0.8,size=(self.n_noise_samples, self.z_len))
        self.label_samples = label_generator(self.n_noise_samples)
        self.build_generator()
        self.build_discriminator()
        self.build_model()


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


    def build_discriminator(self):
        model_in = Input(shape=(self.img_width, self.img_height, self.img_depth))
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

        # architecture of tail stem
        out = Dense(units=1, kernel_initializer='he_uniform')(x)
        embed_labels = Dense(
            K.int_shape(class_in)[-1],
            kernel_initializer='he_uniform'
            )(class_in)
        yh = Multiply()([out, embed_labels])
        model_out = Lambda(lambda x: K.sum(x, axis=-1))(yh)

        self.discriminator = Model([model_in, class_in], model_out)
        self.frozen_discriminator = Network([model_in, class_in], model_out)

        print(self.discriminator.summary())

    def build_model(self):
        d_optimizer = Adam(lr=self.lr, beta_1=0.0, beta_2=0.99)
        g_optimizer = Adam(lr=self.lr, beta_1=0.0, beta_2=0.99)

        # build complete discriminator
        fake_in = Input(shape=(self.img_width, self.img_height, self.img_depth))
        real_in = Input(shape=(self.img_width, self.img_height, self.img_depth))
        class_in = Input(shape=(self.n_classes, ))
        fake_label = self.discriminator([fake_in, class_in])
        real_label = self.discriminator([real_in, class_in])

        self.discriminator_model = Model(
            [real_in, fake_in, class_in],
            [real_label, fake_label, real_label])
        self.discriminator_model.compile(
            d_optimizer,
            loss=[
                nonsat_real_discriminator_loss,
                nonsat_fake_discriminator_loss,
                partial(gradient_penalty_loss, averaged_samples=real_in)
                ]
            )

        self.frozen_discriminator.trainable = False

        # build generator model
        z_in = Input(shape=(self.z_len, ))
        class_in = Input(shape=(self.n_classes, ))
        fake_img = self.generator([z_in, class_in])
        frozen_fake_label = self.frozen_discriminator([fake_img, class_in])

        self.generator_model = Model([z_in, class_in], frozen_fake_label)
        self.generator_model.compile(g_optimizer, nonsat_generator_loss)
        
        print(self.discriminator_model.summary())
        print(self.generator_model.summary())

    def augmentation_pipeline(self, image, p=0.0):
        image = ((image+1.0)*127.5).astype(np.uint8)

        # pixel blitting augmentations
        # x-flip
        if np.random.uniform(low=0.0, high=1.0) < p:
            image = image[:, ::-1, :]
        # 90 degree rotation
        if np.random.uniform(low=0.0, high=1.0) < p:
            direction = random.sample([0, 1, 2, 3], k=1)
            image = np.rot90(image, k=direction[0])
        # we omit translation augmentation because our pipeline does that already

        # general geometric transformations
        # isotropic scaling, aka zoom
        isotropic_scaling = False
        if np.random.uniform(low=0.0, high=1.0) < p:
            magnitude = np.clip(np.random.lognormal(0, (0.2*np.log(2))**2), a_min=0.05, a_max=1.0)
            x_y_length = int(image.shape[0]*magnitude)
            if x_y_length < image.shape[0]:
                y_offset = np.random.randint(low=0, high=img.shape[0]-x_y_length) if (img.shape[0]-x_y_length > 0) else 0
                x_offset = np.random.randint(low=0, high=img.shape[1]-x_y_length) if (img.shape[1]-x_y_length > 0) else 0
                cropped_img = image[y_offset:y_offset+x_y_length, x_offset:x_offset+x_y_length, :]
                cropped_img = Image.fromarray(cropped_img)
                #print(x_y_length, y_offset, x_offset)
                cropped_img = cropped_img.resize((image.shape[1], image.shape[0]))
                image = np.array(cropped_img)
                isotropic_scaling = True
        
        # random rotation
        if np.random.uniform(low=0.0, high=1.0) < p:
            image = Image.fromarray(image)
            image = image.rotate(
                np.random.uniform(low=0, high=360),
                fillcolor=tuple(
                    np.argmax(
                        [np.bincount(x) for x in np.array(image).reshape(-1, 3).T],
                        axis=1
                        ).astype(np.uint8)
                    )
                )
            image = np.array(image).astype(np.uint8)
        # anisotropic scaling
        # in my opinion, applying both isotropic and anisotropic scaling doesn't make sense
        if (np.random.uniform(low=0.0, high=1.0) < p) and (not isotropic_scaling):
            y_magnitude = np.clip(np.random.lognormal(0, (0.2*np.log(2))**2), a_min=0.05, a_max=1.0)
            x_magnitude = np.clip(np.random.lognormal(0, (0.2*np.log(2))**2), a_min=0.05, a_max=1.0)
            y_length = int(image.shape[0]*y_magnitude)
            y_offset = np.random.randint(low=0, high=img.shape[0]-y_length) if (img.shape[0]-y_length > 0) else 0
            x_length = int(image.shape[0]*x_magnitude)
            x_offset = np.random.randint(low=0, high=img.shape[1]-x_length) if (img.shape[1]-x_length > 0) else 0
            #print(y_length, x_length, x_offset, y_offset, x_offset+x_length, y_offset+y_length)
            if (y_magnitude != 1.0) and (x_magnitude != 1.0):
                cropped_img = image[y_offset:y_offset+y_length, x_offset:x_offset+x_length, :]
                cropped_img = Image.fromarray(cropped_img)
                cropped_img = cropped_img.resize((image.shape[1], image.shape[0]))
                image = np.array(cropped_img)
        image = image.astype(np.uint8)

        # color transforms
        # brightness
        if np.random.uniform(low=0.0, high=1.0) < p:
            magnitude = np.random.normal(0, 0.2**2, size=1)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.int32) + np.array([0, 0, int(magnitude*255)]).reshape((1, 1, 3))
            hsv = np.clip(hsv, a_min=0, a_max=255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # contrast
        if np.random.uniform(low=0.0, high=1.0) < p:
            magnitude = np.random.lognormal(1, (0.2*np.log(2))**2)
            image = cv2.addWeighted(image.astype(np.uint8), magnitude, image, 0, 0)
        # luma flip
        if np.random.uniform(low=0.0, high=1.0) < p:
            image = (((-1*((image/127.5)-1))+1)*127.5).astype(np.uint8)
        # hue rotation
        if np.random.uniform(low=0.0, high=1.0) < p:
            magnitude = np.random.uniform(-127, 127)
            hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv_img = hsv_img.astype(np.int32) + np.array([int(magnitude), 0, 0]).reshape((1, 1, 3))
            hsv_img[..., 0] = np.where(hsv_img[..., 0] > 255, hsv_img[..., 0]-255, hsv_img[..., 0])
            hsv_img[..., 0] = np.where(hsv_img[..., 0] < 0, hsv_img[..., 0]+255, hsv_img[..., 0])
            image = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)
        # saturation
        if np.random.uniform(low=0.0, high=1.0) < p:
            magnitude = np.random.lognormal(0, np.log(2)**2)
            hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv_img[..., 1] = hsv_img[..., 1]*magnitude
            hsv_img = np.clip(hsv_img, a_min=0, a_max=255)
            image = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return (image/127.5)-1.0



    ###############################
    ## All our training, etc
    ###############################               

    def train(self, epochs, supplemental_img_dir=None, max_p=0.6, p_increment=0.05):
        card_generator = CardGenerator(
            img_dir=self.training_dir,
            batch_size=self.batch_size,
            n_cpu=self.n_cpu,
            img_dim=self.img_width
            )
        batch_multiplier = 1
        if supplemental_img_dir is not None:
            img_generator = ImgGenerator(
                img_dir=supplemental_img_dir,
                batch_size=self.batch_size,
                n_cpu=self.n_cpu,
                img_dim=self.img_width
                )
            batch_multiplier = 2
        n_batches = card_generator.n_batches*batch_multiplier
        aug_p = 0.0
        d_output_values = []
        for epoch in range(epochs):
            d_loss_accum = []
            g_loss_accum = []

            pbar = tqdm(total=n_batches)
            for batch_i in range(n_batches):
                if (batch_i % 2 == 0:) and (supplemental_img_dir is not None):
                    real_batch, real_labels = img_generator.next()
                else:
                    real_batch, real_labels = card_generator.next()

                d_outputs = self.discriminator.predict(real_batch)
                d_output_values.extend(d_outputs)
                if (batch_i % 4 == 0) and (batch_i > 0):
                    d_sign = np.mean(d_output_values)
                    if d_sign > 0:
                        p += p_increment
                    elif d_sign < 0:
                        p -= p_increment
                    p = np.clip(p, a_min=0.0, a_max=max_p)
                    d_output_values = []

                noise = np.random.normal(0, 1, size=(self.batch_size, self.z_len))
                dummy = np.ones(shape=(self.batch_size,))
                fake_batch = self.generator.predict([noise, real_labels])
                real_batch = np.stack([self.augmentation_pipeline(img) for img in real_batch])
                fake_batch = np.stack([self.augmentation_pipeline(img) for img in fake_batch])
                d_loss = self.discriminator_model.train_on_batch(
                    [real_batch, fake_batch, real_labels],
                    [dummy, dummy, dummy]
                    )
                d_loss_accum.append(d_loss[1]+d_loss[2])
            
                g_loss = self.generator_model.train_on_batch([noise, real_labels], dummy)
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
        predicted_imgs = self.generator.predict([self.noise_samples, self.label_samples])
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
