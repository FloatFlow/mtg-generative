from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer
from keras.layers import Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.layers import Concatenate, Embedding, Flatten, LeakyReLU, Cropping2D, GaussianNoise
from keras.initializers import RandomNormal, VarianceScaling
from model.layers import *


def adainstancenorm_zproj(x, z):
    target_shape = K.int_shape(x)
    gamma = Dense(target_shape[-1],
                  use_bias=True,
                  kernel_initializer=VarianceScaling(scale=1, mode='fan_in', distribution='normal'),
                  bias_initializer='ones')(z)
    gamma = Reshape((1, 1, -1))(gamma)
    beta = Dense(target_shape[-1],
                  use_bias=True,
                  kernel_initializer='zeros')(z) # this has to be init at zero or everything breaks
    beta = Reshape((1, 1, -1))(beta)

    x = AdaInstanceNormalization()([x, beta, gamma])
    return x

def epilogue_block(inputs,
                   style):
    x = NoiseLayer()(inputs)
    x = LeakyReLU(0.2)(x)
    x = adainstancenorm_zproj(x, style)
    return x

def style_generator_block(inputs,
                          style,
                          output_dim,
                          upsample=True,
                          kernel_init='he_normal',
                          noise_init='zeros'):

    # first conv block
    if upsample:
        x = Conv2DTranspose(filters=output_dim,
                            kernel_size=3,
                            strides=2,
                            padding='same',
                            kernel_initializer=kernel_init)(inputs)
        x = LowPassFilter2D()(x)
    else:
        x = Conv2D(filters=output_dim,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer=kernel_init)(inputs)
    x = epilogue_block(x, style)

    # second conv block
    x = Conv2D(filters=output_dim,
               kernel_size=3,
               padding='same',
               kernel_initializer=kernel_init)(x)
    x = epilogue_block(x, style)

    return x

def style_discriminator_block(inputs,
                              output_dim,
                              downsample=True,
                              kernel_init='he_normal',
                              activation='leaky'):
    x = Conv2D(filters=output_dim,
               kernel_size=3,
               padding='same',
               kernel_initializer=kernel_init)(inputs)
    if activation == 'leaky':
        x = LeakyReLU(0.2)(x)
    else:
        x = Activation('relu')(x)
    if downsample:
        x = LowPassFilter2D()(x)
        x = Conv2D(filters=output_dim,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_initializer=kernel_init)(x)
    else:
        x = Conv2D(filters=output_dim,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer=kernel_init)(x)
    if activation == 'leaky':
        x = LeakyReLU(0.2)(x)
    else:
        x = Activation('relu')(x)
    return x

