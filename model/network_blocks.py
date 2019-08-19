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

def style_generator_block(inputs,
                          style,
                          output_dim,
                          upsample=True,
                          kernel_init='he_normal'):

    # first conv block
    if upsample:
        x = UpSampling2D(2, interpolation='nearest')(inputs)
        x = Conv2D(filters=output_dim,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    kernel_initializer=kernel_init)(x)
        #x = Conv2DTranspose(filters=output_dim,
        #                    kernel_size=3,
        #                    strides=2,
        #                    padding='same',
        #                    kernel_initializer=kernel_init)(inputs)
        x = LowPassFilter2D()(x)
    else:
        x = ConvSN2D(filters=output_dim,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer=kernel_init)(inputs)
    x = epilogue_block(x, style, kernel_init=kernel_init)

    # second conv block
    x = ConvSN2D(filters=output_dim,
               kernel_size=3,
               padding='same',
               kernel_initializer=kernel_init)(x)
    x = epilogue_block(x, style, kernel_init=kernel_init)

    return x

def style_discriminator_block(inputs,
                              output_dim,
                              downsample=True,
                              kernel_init='he_normal'):
    x = ConvSN2D(filters=output_dim,
               kernel_size=3,
               padding='same',
               kernel_initializer=kernel_init)(inputs)
    x = LeakyReLU(0.2)(x)

    if downsample:
        x = LowPassFilter2D()(x)
        x = ConvSN2D(filters=output_dim,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_initializer=kernel_init)(x)
    else:
        x = ConvSN2D(filters=output_dim,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer=kernel_init)(x)
    x = LeakyReLU(0.2)(x)

    return x

###############################################################################
## MSGGAN
###############################################################################

def msg_generator_block(x, output_dim, kernel_init, upsample=True):
    if upsample:
        x = UpSampling2D(2, interpolation='bilinear')(x)
    x = Conv2D(filters=output_dim,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer=kernel_init)(x)
    x = LeakyReLU(0.2)(x)
    x = PixelNormalization()(x)
    x = Conv2D(filters=output_dim,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer=kernel_init)(x)
    x = LeakyReLU(0.2)(x)
    x = PixelNormalization()(x)

    img_out = Conv2D(filters=3,
                     kernel_size=1,
                     padding='same',
                     kernel_initializer=kernel_init)(x)
    img_out = Activation('tanh')(img_out)

    return x, img_out

def msg_discriminator_block(x, img_in, output_dim, kernel_init, downsample=True):
    input_ch = K.int_shape(x)[-1]
    img_features = Conv2D(filters=input_ch,
                        kernel_size=1,
                        padding='same',
                        kernel_initializer=kernel_init)(img_in)
    img_features = LeakyReLU(0.2)(img_features)
    x = Concatenate(axis=-1)([x, img_features])
    x = Conv2D(filters=output_dim,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer=kernel_init)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(filters=output_dim,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer=kernel_init)(x)
    x = LeakyReLU(0.2)(x)
    if downsample:
        x = AveragePooling2D(2)(x)
    
    return x

###############################################################################
## MSG StyleGAN
###############################################################################

def msg_style_generator_block(x, style, output_dim, kernel_init, upsample=True):
    # first conv block
    if upsample:
        x = Conv2DTranspose(filters=output_dim,
                            kernel_size=3,
                            strides=2,
                            padding='same',
                            kernel_initializer=kernel_init)(x)
        x = LowPassFilter2D()(x)
    else:
        x = Conv2D(filters=output_dim,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer=kernel_init)(x)
    x = epilogue_block(x, style, kernel_init=kernel_init)

    # second conv block
    x = Conv2D(filters=output_dim,
               kernel_size=3,
               padding='same',
               kernel_initializer=kernel_init)(x)
    x = epilogue_block(x, style, kernel_init=kernel_init)

    img_out = Conv2D(filters=3,
                     kernel_size=1,
                     padding='same',
                     kernel_initializer=kernel_init)(x)
    img_out = Activation('tanh')(img_out)

    return x, img_out

def msg_style_discriminator_block(x, img_in, output_dim, kernel_init, downsample=True):
    input_ch = K.int_shape(x)[-1]
    img_features = Conv2D(filters=input_ch,
                        kernel_size=1,
                        padding='same',
                        kernel_initializer=kernel_init)(img_in)
    x = Concatenate(axis=-1)([x, img_features])
    x = Conv2D(filters=output_dim,
               kernel_size=3,
               padding='same',
               kernel_initializer=kernel_init)(x)
    x = LeakyReLU(0.2)(x)

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
    x = LeakyReLU(0.2)(x)
    
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

###############################################################################
## MSGMiniGAN
###############################################################################

def msg_biggan_generator_block(x, z, ch, upsample=True, kernel_init='he_normal', bias=False, output_img=True):
    # left path
    xl = Lambda(lambda x: x[:,:,:,:ch])(x)
    if upsample:
        xl = UpSampling2D((2,2), interpolation='nearest')(xl)

    # right path
    xr = adainstancenorm_zproj(x, z)
    xr = Activation('relu')(x)
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
    if upsample:
        xr = LowPassFilter2D()(xr)

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

    if output_img:
        img_out = ConvSN2D(filters=3,
                         kernel_size=3,
                         padding='same',
                         kernel_initializer=kernel_init)(x)
        img_out = Activation('tanh')(img_out)
        return x, img_out
    else:
        return x

def msg_biggan_discriminator_block(x, ch, img_in=None, downsample=True, kernel_init='he_normal', bias=False):
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
    if img_in != None:
        input_ch = K.int_shape(x)[-1]
        if K.int_shape(img_in)[1] != K.int_shape(x)[1]:
            img_in = ZeroPadding2D(((1, 0), (1, 0)))(img_in)
        img_features = ConvSN2D(filters=input_ch,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer=kernel_init)(img_in)
        xr = Concatenate(axis=-1)([x, img_features])
    else:
        xr = x

    xr = Activation('relu')(xr)
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
        xr = LowPassFilter2D()(x)
        xr = ConvSN2D(filters=ch,
                      kernel_size=1,
                      strides=2,
                      padding='same',
                      use_bias=bias,
                      kernel_initializer=kernel_init)(xr)
    else:
        xr = ConvSN2D(filters=ch,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      use_bias=bias,
                      kernel_initializer=kernel_init)(xr)

    x = Add()([xl, xr])
    return x

def styleres_generator_block(x, z, ch, upsample=True, kernel_init='he_normal', bias=True, activation='leaky'):
    # left path
    xl = Lambda(lambda x: x[:,:,:,:ch])(x)
    if upsample:
        xl = UpSampling2D((2,2), interpolation='nearest')(xl)

    # right path
    
    xr = epilogue_block(x, z, activation=activation)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = epilogue_block(xr, z, activation=activation)
    if upsample:
        xr = UpSampling2D((2,2), interpolation='nearest')(xr)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)
    #if upsample:
    #    xr = LowPassFilter2D()(xr)

    xr = epilogue_block(xr, z, activation=activation)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = epilogue_block(xr, z, activation=activation)
    xr = ConvSN2D(filters=ch,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    x = Add()([xl, xr])
    return x

def styleres_discriminator_block(x, ch, downsample=True, kernel_init='he_normal', bias=True, activation='leaky'):
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
    if activation == 'leaky':
        xr = LeakyReLU(0.2)(x)
    else:
        xr = Activation('relu')(x)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    if activation == 'leaky':
        xr = LeakyReLU(0.2)(xr)
    else:
        xr = Activation('relu')(xr)
    xr = ConvSN2D(filters=ch//4,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    if activation == 'leaky':
        xr = LeakyReLU(0.2)(xr)
    else:
        xr = Activation('relu')(xr)
    if downsample:
        #xr = LowPassFilter2D()(xr)
        xr = ConvSN2D(filters=ch//4,
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      use_bias=bias,
                      kernel_initializer=kernel_init)(xr)
    else:
        xr = ConvSN2D(filters=ch//4,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      use_bias=bias,
                      kernel_initializer=kernel_init)(xr)

    if activation == 'leaky':
        xr = LeakyReLU(0.2)(xr)
    else:
        xr = Activation('relu')(xr)

    xr = ConvSN2D(filters=ch,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    x = Add()([xl, xr])
    return x

def res_block(x, ch, downsample=True):
    if downsample:
        skip = LowPassFilter2D()(x)
        skip = Conv2D(filters=ch,
               kernel_size=3,
               strides=2,
               padding='same')(skip)
        skip = BatchNormalization()(skip)
    else:
        skip = Conv2D(filters=ch,
                      kernel_size=1,
                      strides=1,
                      padding='same')(x)
        skip = BatchNormalization()(skip)

    if downsample:
        y = LowPassFilter2D()(x)
        y = Conv2D(filters=ch//4,
               kernel_size=3,
               strides=2,
               padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
    else:
        y = Conv2D(filters=ch//4,
                   kernel_size=1,
                   strides=1,
                   padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

    y = Conv2D(filters=ch//4,
               kernel_size=3,
               strides=1,
               padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=ch,
           kernel_size=3,
           strides=1,
           padding='same')(y)
    y = BatchNormalization()(y)

    output = Add()([y, skip])
    output = Activation('relu')(output)
    return output

