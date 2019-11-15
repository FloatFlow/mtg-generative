from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer
from keras.layers import Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.layers import Concatenate, Embedding, Flatten, LeakyReLU, Cropping2D, GaussianNoise
from keras.initializers import RandomNormal, VarianceScaling
from model.layers import *


def adainstancenorm_zproj(x, z):
    target_shape = K.int_shape(x)
    gamma = Dense(units=target_shape[-1],
                  use_bias=True,
                  kernel_initializer='zeros',
                  bias_initializer='ones')(z)
    gamma = Reshape((1, 1, -1))(gamma)
    beta = Dense(units=target_shape[-1],
                  use_bias=True,
                  kernel_initializer='zeros')(z) # this has to be init at zero or everything breaks
    beta = Reshape((1, 1, -1))(beta)

    x = AdaInstanceNormalization()([x, beta, gamma])
    return x

###############################################################################
## StyleGAN
###############################################################################

def epilogue_block(inputs,
                   style,
                   activation='leaky'):
    x = NoiseLayer()(inputs)
    if activation == 'leaky':
        x = LeakyReLU(0.2)(x)
    else:
        x = Activation('relu')(x)
    x = adainstancenorm_zproj(x, style)
    return x

def style_generator_block(
    inputs,
    style,
    output_dim,
    upsample=True,
    kernel_init='he_normal'
    ):

    # first conv block
    if upsample:
        x = UpSampling2D(2, interpolation='nearest')(inputs)
        x = Conv2DTranspose(
            filters=output_dim,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=kernel_init
            )(inputs)
        x = LowPassFilter2D()(x)
    else:
        x = Conv2D(
            filters=output_dim,
            kernel_size=3,
            padding='same',
            kernel_initializer=kernel_init
            )(inputs)
    x = epilogue_block(x, style, kernel_init=kernel_init)

    # second conv block
    x = Conv2D(
        filters=output_dim,
        kernel_size=3,
        padding='same',
        kernel_initializer=kernel_init
        )(x)
    x = epilogue_block(x, style, kernel_init=kernel_init)

    return x

def style_discriminator_block(
    inputs,
    output_dim,
    downsample=True,
    kernel_init='he_normal',
    activation='relu'
    ):
    x = Conv2D(
        filters=output_dim,
        kernel_size=3,
        padding='same',
        kernel_initializer=kernel_init
        )(inputs)
    if activation == 'relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(0.2)(x)

    if downsample:
        x = LowPassFilter2D()(x)
        x = Conv2D(
            filters=output_dim,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=kernel_init
            )(x)
    else:
        x = Conv2D(
            filters=output_dim,
            kernel_size=3,
            padding='same',
            kernel_initializer=kernel_init
            )(x)
    if activation == 'relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(0.2)(x)

    return x

def style_generator_block(
    inputs,
    style,
    output_dim,
    upsample=True,
    kernel_init='he_normal'
    ):

    # first conv block
    if upsample:
        x = UpSampling2D(2, interpolation='nearest')(inputs)
        x = Conv2DTranspose(
            filters=output_dim,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=kernel_init
            )(inputs)
        x = LowPassFilter2D()(x)
    else:
        x = Conv2D(
            filters=output_dim,
            kernel_size=3,
            padding='same',
            kernel_initializer=kernel_init
            )(inputs)
    x = epilogue_block(x, style, kernel_init=kernel_init)

    # second conv block
    x = Conv2D(
        filters=output_dim,
        kernel_size=3,
        padding='same',
        kernel_initializer=kernel_init
        )(x)
    x = epilogue_block(x, style, kernel_init=kernel_init)

    return x

def style_decoder_block(
    inputs,
    output_dim,
    upsample=True,
    kernel_init='he_normal',
    activation='relu'
    ):

    # first conv block
    if upsample:
        x = UpSampling2D(2, interpolation='nearest')(inputs)
        x = Conv2DTranspose(
            filters=output_dim,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=kernel_init
            )(inputs)
        x = LowPassFilter2D()(x)
    else:
        x = Conv2D(
            filters=output_dim,
            kernel_size=3,
            padding='same',
            kernel_initializer=kernel_init
            )(inputs)
    if activation == 'leaky':
        x = LeakyReLU(0.2)(x)
    else:
        x = Activation('relu')(x)
    x = InstanceNormalization()(x)

    # second conv block
    x = Conv2D(
        filters=output_dim,
        kernel_size=3,
        padding='same',
        kernel_initializer=kernel_init
        )(x)
    if activation == 'leaky':
        x = LeakyReLU(0.2)(x)
    else:
        x = Activation('relu')(x)
    x = InstanceNormalization()(x)
    return x


###############################################################################
## MiniGAN
###############################################################################

def deep_biggan_generator_block(x, z, ch, upsample=True, kernel_init='he_normal', bias=False):
    # left path
    xl = Lambda(lambda x: x[:,:,:,:ch])(x)
    if upsample:
        xl = UpSampling2D((2,2), interpolation='nearest')(xl)

    # right path
    
    xr = adainstancenorm_zproj(x, z)
    xr = Activation('relu')(xr)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = adainstancenorm_zproj(xr, z)
    xr = Activation('relu')(xr)
    if upsample:
        xr = UpSampling2D((2,2), interpolation='nearest')(xr)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = adainstancenorm_zproj(xr, z)
    xr = Activation('relu')(xr)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = adainstancenorm_zproj(xr, z)
    xr = Activation('relu')(xr)
    xr = ConvSN2D(filters=ch,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    x = Add()([xl, xr])
    return x

def deep_biggan_discriminator_block(x, ch, downsample=True, kernel_init='he_normal', bias=False):
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
                        use_bias=bias,
                        kernel_initializer=kernel_init)(xl)
        xl = Concatenate()([xl, xl_l])

    # right path
    xr = Activation('relu')(x)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = Activation('relu')(xr)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = Activation('relu')(xr)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = Activation('relu')(xr)
    if downsample:
        xr = AveragePooling2D((2,2))(xr)
    xr = ConvSN2D(filters=ch,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    x = Add()([xl, xr])
    return x

def deep_simple_biggan_generator_block(x, ch, upsample=True, kernel_init='he_normal', bias=True):
    # left path
    xl = Lambda(lambda x: x[:, :, :, :ch])(x)
    if upsample:
        xl = UpSampling2D((2,2), interpolation='nearest')(xl)

    # right path
    xr = Activation('relu')(x)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = Activation('relu')(xr)
    if upsample:
        xr = UpSampling2D((2,2), interpolation='nearest')(xr)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = Activation('relu')(xr)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = Activation('relu')(xr)
    xr = ConvSN2D(filters=ch,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    x = Add()([xl, xr])
    return x

