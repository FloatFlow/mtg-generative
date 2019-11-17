from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer
from keras.layers import Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.layers import Concatenate, Embedding, Flatten, LeakyReLU, Cropping2D, GaussianNoise
from keras.initializers import RandomNormal, VarianceScaling
from model.layers import *


###############################################################################
## StyleGAN
###############################################################################

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

###############################################################################
## Res Blocks
###############################################################################

def resblock_decoder(
    inputs,
    output_dim,
    upsample=True,
    kernel_init=VarianceScaling(np.sqrt(2))
    ):
    if upsample:
        x_skip = Conv2DTranspose(
            filters=output_dim,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=kernel_init
            )(inputs)
    else:
        x_skip = Conv2D(
            filters=output_dim,
            kernel_size=1,
            padding='same',
            kernel_initializer=kernel_init
            )(inputs)

    # first conv block
    if upsample:
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
    x = Activation('relu')(x)
    #x = InstanceNormalization()(x)

    # second conv block
    x = Conv2D(
        filters=output_dim,
        kernel_size=3,
        padding='same',
        kernel_initializer=kernel_init
        )(x)
    x = Activation('relu')(x)
    #x = InstanceNormalization()(x)

    x = Add()([x_skip, x])
    return x

def resblock_encoder(
    inputs,
    output_dim,
    downsample=True,
    kernel_init=VarianceScaling(np.sqrt(2))
    ):
    if downsample:
        x_skip = Conv2D(
            filters=output_dim,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=kernel_init
            )(inputs)
    else:
        x_skip = Conv2D(
            filters=output_dim,
            kernel_size=1,
            padding='same',
            kernel_initializer=kernel_init
            )(inputs)

    x = Conv2D(
        filters=output_dim,
        kernel_size=3,
        padding='same',
        kernel_initializer=kernel_init
        )(inputs)
    x = Activation('relu')(x)

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
    x = Activation('relu')(x)

    x = Add()([x, x_skip])
    return x

###############################################################################
## PixelCNN
## Adopted from https://github.com/suga93/pixelcnn_keras/blob/master/core/layers.py
###############################################################################

def masked_conv(x, filter_size, stack_type, mask_type='B'):
    if stack_type == 'vertical':
        res = ZeroPadding2D(
            padding=((filter_size[0]//2, 0), (filter_size[1]//2, filter_size[1]//2))
            )(x)
        res = Conv2D(
            filters=2*self.n_filters,
            kernel_size=(filter_size[0]//2+1, filter_size[1]),
            padding='valid'
            )(res)
    elif stack_type == 'horizontal':
        res = ZeroPadding2D(padding=((0, 0), (filter_size[1]//2, 0)))(x)
        if mask_type == 'A':
            res = Conv2D(
                filters=2*self.n_filters,
                kernel_size=(1, filter_size[1]//2),
                padding='valid'
                )(res)
        else:
            res = Conv2D(
                filters=2*self.n_filters,
                kernel_size=(1, filter_size[1]//2+1),
                padding='valid'
                )(res)
    return res

def shift_down(x):
    x_shape = K.int_shape(x)
    x = ZeroPadding2D(padding=((1, 0), (0, 0)))(x)
    x = Lambda(lambda x: x[:, :x_shape[1], ...])(x)
    return x

def feed_v_map(x, n_filters=64):
    ### shifting down feature maps
    x = shift_down(x)
    x = Conv2D(
        filters=2*n_filters,
        kernel_size=1,
        padding='valid'
        )(x)
    return x

def intro_pixelcnn_layer(x, filter_size=(3, 3), n_filters=64, h=None):
    # first PixelCNN layer
    ### (kxk) masked convolution can be achieved by (k//2+1, k) convolution and padding.
    v_masked_map = masked_conv(
        x,
        filter_size=filter_size,
        stack_type='vertical'
        )
    ### (i-1)-th vertical activation maps into the i-th hirizontal stack. (if i==0, vertical activation maps == input images)
    v_feed_map = feed_v_map(v_masked_map, n_filters)
    v_stack_out = GatedCNN(
        n_filters=n_filters,
        stack_type='vertical',
        v_map=None,
        h=h
        )(v_masked_map)
    ### (1xk) masked convolution can be achieved by (1 x k//2+1) convolution and padding.
    h_masked_map = masked_conv(
        x,
        filter_size=filter_size,
        stack_type='horizontal',
        mask_type='A'
        )
    ### Mask A is applied to the first layer (achieved by cropping), and v_feed_maps are merged. 
    h_stack_out = GatedCNN(
        n_filters=n_filters,
        stack_type='horizontal',
        v_map=v_feed_map,
        h=h,
        crop_right=True
        )(h_masked_map)
    ### not residual connection in the first layer.
    h_stack_out = Conv2D(
        filters=n_filters,
        kernel_size=1,
        paddomg='valid'
        )(h_stack_out)
    return (v_stack_out, h_stack_out)

def pixelcnn_layer(v_stack_in, h_stack_in, filter_size=(3, 3), n_filters=64, h=None):
    v_masked_map = masked_conv(
        v_stack_in,
        filter_size, 
        stack_type='vertical')
    v_feed_map = feed_v_map(v_masked_map)
    v_stack_out = GatedCNN(
        n_filters,
        stack_type='vertical',
        v_map=None,
        h=h
        )(v_masked_map)
    ### for residual connection
    h_masked_map = masked_conv(h_stack_in, filter_size, 'horizontal')
    ### Mask B is applied to the subsequent layers.
    h_stack_out = GatedCNN(
        n_filters,
        'horizontal',
        v_map=v_feed_map,
        h=self.h
        )(h_masked_map)
    h_stack_out = Conv2D(
        filters=n_filters,
        kernel_size=1,
        padding='valid'
        )(h_stack_out)
    ### residual connection
    h_stack_out = Add()([h_stack_in, h_stack_out])
    return (v_stack_out, h_stack_out)

