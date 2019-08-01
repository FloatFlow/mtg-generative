from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer
from keras.layers import Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.layers import Concatenate, Embedding, Flatten, LeakyReLU, Cropping2D
from keras.initializers import RandomNormal
from model.layers import *


def adainstancenorm_zproj(x, z, kernel_init='he_normal'):
    # note bigGAN init gamma and beta kernels as N(0, 0.02)
    target_shape = K.int_shape(x)
    gamma = Dense(target_shape[-1],
                  use_bias=True,
                  kernel_initializer=kernel_init,
                  bias_initializer='ones')(z)
    gamma = Reshape((1, 1, -1))(gamma)
    beta = Dense(target_shape[-1],
                  use_bias=True,
                  kernel_initializer='zeros')(z)
    beta = Reshape((1, 1, -1))(beta)

    x = AdaInstanceNormalization()([x, beta, gamma])
    return x


def style_generator_block(inputs,
                          style,
                          noise,
                          output_dim,
                          upsample=True,
                          kernel_init='he_normal',
                          noise_init='zeros'):
    if upsample:
        input_shape = K.int_shape(inputs)[1]*2
    else:
        input_shape = K.int_shape(inputs)[1]
    noise_shape = K.int_shape(noise)[1]
    pad_size = (noise_shape - input_shape) // 2

    # first conv block
    
    if upsample:
        x = EqualizedConv2DTranspose(filters=output_dim,
                                     kernel_size=3,
                                     strides=2,
                                     padding='same',
                                     kernel_initializer=kernel_init)(inputs)
        x = LowPassFilter2D()(x)
    else:
        x = EqualizedConv2D(filters=output_dim,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer=kernel_init)(x)
    fuzz = Cropping2D(pad_size)(noise)
    fuzz = EqualizedConv2D(filters=output_dim,
                  kernel_size=1,
                  padding='same',
                  kernel_initializer=noise_init)(fuzz)

    x = adainstancenorm_zproj(x, style, kernel_init=kernel_init)
    x = Add()([x, fuzz])
    x = LeakyReLU(0.2)(x)

    # second conv block
    
    x = EqualizedConv2D(filters=output_dim,
               kernel_size=3,
               padding='same',
               kernel_initializer=kernel_init)(x)
    fuzz = Cropping2D(pad_size)(noise)
    fuzz = EqualizedConv2D(filters=output_dim,
                  kernel_size=1,
                  padding='same',
                  kernel_initializer=noise_init)(fuzz)
    x = adainstancenorm_zproj(x, style, kernel_init=kernel_init)
    x = Add()([x, fuzz])
    x = LeakyReLU(0.2)(x)

    return x

def style_discriminator_block(inputs,
                              output_dim,
                              downsample=True,
                              kernel_init='he_normal'):
    x = EqualizedConv2D(filters=output_dim,
               kernel_size=3,
               padding='same',
               kernel_initializer=kernel_init)(inputs)
    x = LeakyReLU(0.2)(x)
    if downsample:
        x = LowPassFilter2D()(x)
        x = EqualizedConv2D(filters=output_dim,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_initializer=kernel_init)(x)
    else:
        x = EqualizedConv2D(filters=output_dim,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer=kernel_init)(x)
    x = LeakyReLU(0.2)(x)
    return x
