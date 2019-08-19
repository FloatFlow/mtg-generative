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
        self.kernel_init = 'he_normal'
        self.build_encoder()
        self.build_decoder()


    ###############################
    ## All our architecture
    ###############################

    def build_encoder(self, ch=16):
        encoder_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))

        x = ConvSN2D(filters=ch,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     use_bias=True,
                     kernel_initializer=self.kernel_init)(encoder_in)
        ch *= 2
        x = deep_biggan_discriminator_block(x, ch, downsample=True, bias=True) #128x32

        ch *= 2
        x = deep_biggan_discriminator_block(x, ch, downsample=True, bias=True) #64x64

        #x = Attention_2SN(ch)(x) #64x64

        quantized64, loss64 = VectorQuantizerEMA(num_embeddings=64, embedding_dim=64)(x)
        ch *= 2
        x = deep_biggan_discriminator_block(x, ch, downsample=True, bias=True) #32x128

        quantized32, loss32 = VectorQuantizerEMA(num_embeddings=128, embedding_dim=128)(x)

        loss = Add()([loss64, loss32])
        #perplexity = Lambda(lambda x: K.sum(x[0], x[1]))([perplexity64, perplexity32])

        self.encoder = Model(encoder_in, [quantized64, quantized32, loss])
        print(self.encoder.summary())
        with open('{}_architecture.txt'.format(self.name), 'w') as f:
            self.encoder.summary(print_fn=lambda x: f.write(x + '\n'))

    def build_decoder(self, ch=128):
        img_in32 = Input(shape=(self.img_dim_x//8, self.img_dim_y//8, ch))

        x = deep_simple_biggan_generator_block(img_in32, ch, upsample=False) #32x128
        ch = ch//2
        x = deep_simple_biggan_generator_block(x, ch, upsample=True) #64x64

        #x = Attention_2SN(ch)(x) #64x64
        
        img_in64 = Input(shape=(self.img_dim_x//4, self.img_dim_y//4, ch))

        x = Concatenate(axis=-1)([x, img_in64])
        ch = ch//2
        x = deep_simple_biggan_generator_block(x, ch, upsample=True) #128x32

        ch = ch//2
        x = deep_simple_biggan_generator_block(x, ch, upsample=True) #256x16

        model_out = ConvSN2D(filters=3,
                             kernel_size=3,
                             strides=1,
                             padding='same',
                             use_bias=True,
                             kernel_initializer=self.kernel_init,
                             activation='tanh')(x)

        self.decoder = Model([img_in32, img_in64], model_out)   
        print(self.decoder.summary())
        with open('{}_architecture.txt'.format(self.name), 'a') as f:
            self.decoder.summary(print_fn=lambda x: f.write(x + '\n'))

    def build_model(self):
        pass


    ###############################
    ## All our training, etc
    ###############################               

    def train(self, epochs):
        sess = tf.Session()
        K.set_session(sess)

        card_generator = CardGenerator(img_dir=self.training_dir,
                                       batch_size=self.batch_size,
                                       n_cpu=self.n_cpu,
                                       img_dim=self.img_dim_x)
        real_batch, real_labels = card_generator.next()
        self.selected_samples = real_batch[:16]

        #img_in = Input(shape=(self.img_dim_x, self.img_dim_y, self.img_depth))
        img_in = tf.placeholder(tf.float32, shape=(None, self.img_dim_x, self.img_dim_y, self.img_depth))
        quantized64, quantized32, commitment_loss  = self.encoder(img_in)
        reconstructed_img = self.decoder([quantized32, quantized64])
        real_tensor = tf.placeholder(tf.float32, shape=(None, self.img_dim_x, self.img_dim_y, self.img_depth))
        loss = vq_reconstruction_loss(real_tensor, img_in, commitment_loss)
        tf.losses.add_loss(loss)
        train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        # Initialize all variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Run training loop
        with sess.as_default():                
            n_batches = card_generator.n_batches
            for epoch in range(epochs):
                loss_accum = []

                pbar = tqdm(total=n_batches)
                for batch_i in range(n_batches):
                    real_batch, real_labels = card_generator.next()
                    
                    train_step.run(feed_dict={img_in: real_batch,
                                              real_tensor: real_batch})
                    #loss = K.eval(tf.losses.get_losses()[0])
                    #loss = self.model.train_on_batch(real_batch, real_batch)

                    pbar.update()
                pbar.close()

                print('{}/{} ----> loss: {}'.format(epoch, 
                                                    epochs, 
                                                    loss))

                if epoch % self.save_freq == 0:
                    self.reconstruction_validation(epoch)
                    self.save_model_weights(epoch, loss)

                card_generator.shuffle()

            card_generator.end()                


    def reconstruction_validation(self, epoch):
        print('Generating Images...')
        if not os.path.isdir(self.validation_dir):
            os.mkdir(self.validation_dir)
        reconstructed_imgs = self.generator.predict(self.selected_samples)
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


    def noise_validation_wlabel(self):
        pass

    def predict_noise_wlabel_testing(self, label):
        pass

    def save_model_weights(self, epoch, loss):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        decoder_savename = os.path.join(self.checkpoint_dir, 'vqvae_decoder_weights_{}_{:.3f}.h5'.format(epoch, loss))
        encoder_savename = os.path.join(self.checkpoint_dir, 'vqvae_encoder_weights_{}_{:.3f}.h5'.format(epoch, loss))
        self.decoder.save_weights(decoder_savename)
        self.encoder.save_weights(encoder_savename)

    def load_model_weights(self, d_weight_path, e_weight_path):
        self.decoder.load_weights(d_weight_path, by_name=True)
        self.encoder.load_weights(e_weight_path, by_name=True)
