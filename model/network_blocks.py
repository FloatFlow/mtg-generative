from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer
from keras.layers import Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.layers import Concatenate, Embedding, Flatten, LeakyReLU, Cropping2D, GaussianNoise
from keras.initializers import RandomNormal, VarianceScaling
from model.layers import *


def adainstancenorm_zproj(x, z, kernel_init):
    target_shape = K.int_shape(x)
    gamma = EqualizedDense(target_shape[-1],
                  gain=1,
                  use_bias=True,
                  kernel_initializer=kernel_init,
                  bias_initializer='ones')(z)
    gamma = Reshape((1, 1, -1))(gamma)
    beta = EqualizedDense(target_shape[-1],
                  gain=1,
                  use_bias=True,
                  kernel_initializer='zeros')(z) # this has to be init at zero or everything breaks
    beta = Reshape((1, 1, -1))(beta)

    x = AdaInstanceNormalization()([x, beta, gamma])
    return x

def epilogue_block(inputs,
                   style,
                   kernel_init):
    x = NoiseLayer()(inputs)
    x = LeakyReLU(0.2)(x)
    x = adainstancenorm_zproj(x, style, kernel_init)
    return x

def style_generator_block(inputs,
                          style,
                          output_dim,
                          upsample=True,
                          kernel_init='he_normal',
                          noise_init='zeros'):

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
                   kernel_initializer=kernel_init)(inputs)
    x = epilogue_block(x, style)

    # second conv block
    x = EqualizedConv2D(filters=output_dim,
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
    x = EqualizedConv2D(filters=output_dim,
               kernel_size=3,
               padding='same',
               kernel_initializer=kernel_init)(inputs)
    if activation == 'leaky':
        x = LeakyReLU(0.2)(x)
    else:
        x = Activation('relu')(x)
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
    if activation == 'leaky':
        x = LeakyReLU(0.2)(x)
    else:
        x = Activation('relu')(x)
    return x

def msg_generator_block(x, output_dim, kernel_init, upsample=True):
    if upsample:
        x = UpSampling2D(2, interpolation='bilinear')(x)
    x = EqualizedConv2D(filters=output_dim,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer=kernel_init)(x)
    x = LeakyReLU(0.2)(x)
    x = PixelNormalization()(x)
    x = EqualizedConv2D(filters=output_dim,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer=kernel_init)(x)
    x = LeakyReLU(0.2)(x)
    x = PixelNormalization()(x)

    img_out = EqualizedConv2D(filters=3,
                              kernel_size=1,
                              padding='same',
                              kernel_initializer=kernel_init)(x)
    img_out = Activation('tanh')(img_out)

    return x, img_out

def msg_discriminator_block(x, img_in, output_dim, kernel_init, downsample=True):
    input_ch = K.int_shape(x)[-1]
    img_features = EqualizedConv2D(filters=input_ch,
                        kernel_size=1,
                        padding='same',
                        kernel_initializer=kernel_init)(img_in)
    img_features = LeakyReLU(0.2)(img_features)
    x = Concatenate(axis=-1)([x, img_features])
    x = EqualizedConv2D(filters=output_dim,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer=kernel_init)(x)
    x = LeakyReLU(0.2)(x)
    
    x = EqualizedConv2D(filters=output_dim,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer=kernel_init)(x)
    x = LeakyReLU(0.2)(x)
    if downsample:
        x = AveragePooling2D(2)(x)
    
    return x

