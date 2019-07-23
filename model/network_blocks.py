from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer
from keras.layers import Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.layers import Concatenate, Embedding, Flatten
from model.layers import *

def adainstancenorm_zproj(x, z, kernel_init='he_normal'):
    # note bigGAN init gamma and beta kernels as N(0, 0.02)
    target_shape = K.int_shape(x)
    gamma = Dense(target_shape[-1],
                  use_bias=True,
                  kernel_initializer=kernel_init)(z)
    gamma = Reshape((1, 1, -1))(gamma)
    beta = Dense(target_shape[-1],
                  use_bias=True,
                  kernel_initializer=kernel_init)(z)
    beta = Reshape((1, 1, -1))(beta)

    x = AdaInstanceNormalization()([x, beta, gamma])
    return x

def deep_resblock_up(x,
                     z,
                     ch,
                     squeeze=4,
                     kernel_init='he_normal',
                     upsample=True):
    # left path
    xl = Lambda(lambda x: x[:,:,:,:ch])(x)
    if upsample:
        xl = UpSampling2D((2,2), interpolation='nearest')(xl)

    # right path
    
    xr = adainstancenorm_zproj(x, z)
    xr = Activation('relu')(x)
    xr = ConvSN2D(filters=ch//squeeze,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=True,
                  kernel_initializer=kernel_init)(xr)

    xr = adainstancenorm_zproj(xr, z)
    xr = Activation('relu')(xr)
    if upsample:
        xr = UpSampling2D((2,2), interpolation='nearest')(xr)
    xr = ConvSN2D(filters=ch//squeeze,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=True,
                  kernel_initializer=kernel_init)(xr)

    xr = adainstancenorm_zproj(xr, z)
    xr = Activation('relu')(xr)
    xr = ConvSN2D(filters=ch//squeeze,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=True,
                  kernel_initializer=kernel_init)(xr)

    xr = adainstancenorm_zproj(xr, z)
    xr = Activation('relu')(xr)
    xr = ConvSN2D(filters=ch,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=True,
                  kernel_initializer=kernel_init)(xr)

    x = Add()([xl, xr])
    return x


def deep_resblock_down(x,
                       ch,
                       kernel_init='he_normal',
                       squeeze=4,
                       downsample=True):
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
                        use_bias=True,
                        kernel_initializer=kernel_init)(xl)
        xl = Concatenate()([xl, xl_l])

    # right path
    xr = Activation('relu')(x)
    xr = ConvSN2D(filters=ch//squeeze,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=True,
                  kernel_initializer=kernel_init)(xr)

    xr = Activation('relu')(xr)
    xr = ConvSN2D(filters=ch//squeeze,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=True,
                  kernel_initializer=kernel_init)(xr)

    xr = Activation('relu')(xr)
    xr = ConvSN2D(filters=ch//squeeze,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=True,
                  kernel_initializer=kernel_init)(xr)

    xr = Activation('relu')(xr)
    if downsample:
        xr = AveragePooling2D((2,2))(xr)
    xr = ConvSN2D(filters=ch,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=True,
                  kernel_initializer=kernel_init)(xr)

    x = Add()([xl, xr])
    return x
