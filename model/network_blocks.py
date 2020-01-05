from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer, \
                         Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input, \
                         Concatenate, Embedding, Flatten, LeakyReLU, Cropping2D, GaussianNoise, \
                         ZeroPadding2D, add
from keras.initializers import RandomNormal, VarianceScaling
from model.layers import *

###############################################################################
## SinGAN
###############################################################################

def singan_generator_block(inputs, noise, filters=32):
    x = Add()([inputs, noise])
    for _ in range(5):
        x = Conv2D(
            filters=filters,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_uniform'
            )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
    x = Conv2D(
        filters=3,
        kernel_size=1,
        padding='same',
        kernel_initializer='he_uniform'
        )(x)
    x = Add()([inputs, x])
    x = Activation('tanh')(x)
    return x

def singan_discriminator_block(inputs, filters=32):
    x = inputs
    for _ in range(5):
        x = Conv2D(
            filters=filters,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_uniform'
            )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
    x = Conv2D(
        filters=1,
        kernel_size=1,
        padding='same',
        kernel_initializer='he_uniform'
        )(x)
    x = Flatten()(x)
    x = Dense(1, kernel_initializer='he_uniform')(x)
    return x

###############################################################################
## StyleGAN
###############################################################################

def style2_generator_layer(
    inputs,
    style,
    output_dim,
    upsample=False,
    kernel_init='he_uniform'
    ):
    style = Dense(
        K.int_shape(inputs)[-1],
        kernel_initializer=kernel_init,
        bias_initializer='ones'
        )(style)
    x = ModulatedConv2D(
        filters=output_dim,
        kernel_size=3,
        padding='same',
        upsample=upsample
        )([inputs, style])
    x = NoiseLayer()(x)
    x = Bias()(x)
    x = LeakyReLU(0.2)(x)
    return x

def to_rgb(inputs, style):
    kernel_init = VarianceScaling(200/K.int_shape(inputs)[2])
    style = Dense(
        K.int_shape(inputs)[-1],
        kernel_initializer=kernel_init,
        bias_initializer='ones'
        )(style)
    style = LeakyReLU(0.2)(style)
    x = ModulatedConv2D(
        filters=3,
        kernel_size=1,
        padding='same',
        demodulate=False,
        kernel_initializer=kernel_init
        )([inputs, style])
    x = Bias()(x)
    x = Activation('tanh')(x)
    return x

def style2_discriminator_block(inputs, output_dim, downsample=False, kernel_init='he_uniform'):
    skip = inputs
    skip = AveragePooling2D(2)(skip)
    if K.int_shape(skip)[-1] != output_dim:
        skip = Conv2D(
            filters=output_dim,
            kernel_size=1,
            padding='same',
            kernel_initializer=kernel_init
            )(skip)

    x = Conv2D(
        filters=output_dim,
        kernel_size=3,
        padding='same',
        kernel_initializer=kernel_init
        )(inputs)
    x = LeakyReLU(0.2)(x)
    x = AveragePooling2D(2)(x)
    x = Conv2D(
        filters=output_dim,
        kernel_size=3,
        padding='same',
        kernel_initializer=kernel_init
        )(x)
    x = LeakyReLU(0.2)(x)

    x = Lambda(lambda x: add(x) * (1/np.sqrt(2)))([x, skip])
    return x

###############################################################################
## MiniGAN
###############################################################################

def deep_biggan_generator_block(
    x,
    z,
    ch,
    upsample=True,
    kernel_init='he_uniform',
    bias=True,
    activation='relu',
    scaling_factor=4
    ):
    # left path
    xl = Lambda(lambda x: x[..., :ch])(x)
    if upsample:
        xl = UpSampling2D((2,2), interpolation='nearest')(xl)

    # right path
    
    xr = adainstancenorm_zproj(x, z)
    if activation == 'relu':
        xr = Activation('relu')(xr)
    else:
        xr = LeakyReLU(0.2)(xr)
    xr = ConvSN2D(filters=ch//scaling_factor,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = adainstancenorm_zproj(xr, z)
    if activation == 'relu':
        xr = Activation('relu')(xr)
    else:
        xr = LeakyReLU(0.2)(xr)
    if upsample:
        xr = UpSampling2D((2,2), interpolation='nearest')(xr)
    xr = ConvSN2D(filters=ch//scaling_factor,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = adainstancenorm_zproj(xr, z)
    if activation == 'relu':
        xr = Activation('relu')(xr)
    else:
        xr = LeakyReLU(0.2)(xr)
    xr = ConvSN2D(filters=ch//scaling_factor,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    xr = adainstancenorm_zproj(xr, z)
    if activation == 'relu':
        xr = Activation('relu')(xr)
    else:
        xr = LeakyReLU(0.2)(xr)
    xr = ConvSN2D(filters=ch,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    x = Add()([xl, xr])
    return x

def deep_biggan_discriminator_block(
    x,
    ch,
    downsample=True,
    kernel_init='he_uniform',
    bias=True,
    activation='relu',
    scaling_factor=4
    ):
    # left path
    if downsample:
        xl = AveragePooling2D(2)(x)
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
    if activation == 'relu':
        xr = Activation('relu')(x)
    else:
        xr = LeakyReLU(0.2)(x)
    xr = ConvSN2D(filters=ch//scaling_factor,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    if activation == 'relu':
        xr = Activation('relu')(xr)
    else:
        xr = LeakyReLU(0.2)(xr)
    xr = ConvSN2D(filters=ch//scaling_factor,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    if activation == 'relu':
        xr = Activation('relu')(xr)
    else:
        xr = LeakyReLU(0.2)(xr)
    xr = ConvSN2D(filters=ch//scaling_factor,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  use_bias=bias,
                  kernel_initializer=kernel_init)(xr)

    if activation == 'relu':
        xr = Activation('relu')(xr)
    else:
        xr = LeakyReLU(0.2)(xr)
    if downsample:
        xr = AveragePooling2D(2)(xr)
    xr = ConvSN2D(filters=K.int_shape(xl)[-1],
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

def resblock(
    inputs,
    output_dim,
    kernel_init=VarianceScaling(np.sqrt(2))
    ):
    if K.int_shape(inputs)[-1] != output_dim:
        skip = Conv2D(
            filters=output_dim,
            kernel_size=1,
            padding='same',
            kernel_initializer=kernel_init
            )(inputs)
    else:
        skip = inputs

    x = Activation('relu')(inputs)
    x = Conv2D(
        filters=output_dim//2,
        kernel_size=3,
        padding='same',
        kernel_initializer=kernel_init
        )(x)

    x = Activation('relu')(x)
    x = Conv2D(
        filters=output_dim,
        kernel_size=1,
        padding='same',
        kernel_initializer=kernel_init
        )(x)

    x = Add()([x, skip])
    return x

###############################################################################
## PixelCNN
## Adopted from https://github.com/suga93/pixelcnn_keras/blob/master/core/layers.py
###############################################################################

def context_projection(stack, context):
    if len(K.int_shape(context)) == 2:
        context = Dense(K.int_shape(stack)[-1])(context)
        context = Reshape((1, 1, -1))(context)
    else:
        x_padding = K.int_shape(stack)[-2] - K.int_shape(context)[-2]
        y_padding = K.int_shape(stack)[-3] - K.int_shape(context)[-3]
        context = ZeroPadding2D(((0, y_padding), (0, x_padding)))(context)
        context = Conv2D(
            filters=K.int_shape(stack)[-1],
            kernel_size=1,
            padding='same'
            )(context)
    contextualized_stack = Add()([stack, context])
    return contextualized_stack

def gated_activation(inputs):
    ch_shape = K.int_shape(inputs)[-1]
    x = Lambda(
        lambda x: K.tanh(x[..., :ch_shape//2]) * K.sigmoid(x[..., ch_shape//2:])
        )(inputs)
    return x
    
def gated_masked_conv2d(v_stack_in, h_stack_in, out_dim, kernel, mask='b', residual=True, context=None, use_context=False):
    """
    Basic Gated-PixelCNN block. 
    This is an improvement over PixelRNN to avoid "blind spots", i.e. pixels missingt from the
    field of view. It works by having two parallel stacks, for the vertical and horizontal direction, 
    each being masked  to only see the appropriate context pixels.
    Adapted from https://www.kaggle.com/ameroyer/keras-vq-vae-for-image-generation
    """
    kernel_size = (kernel // 2 + 1, kernel)
    padding = (kernel // 2, kernel // 2)
        
    v_stack = ZeroPadding2D(padding=padding)(v_stack_in)
    v_stack = MaskedConv2D(
        kernel_size=kernel_size,
        out_dim=out_dim*2, 
        direction="v",
        mode=mask
        )(v_stack)
    v_stack = Lambda(lambda x: x[:, :K.int_shape(v_stack_in)[-3], :K.int_shape(v_stack_in)[-2], :])(v_stack)
    if use_context:
        v_stack = context_projection(v_stack, context)
    v_stack_out = gated_activation(v_stack)
    
    kernel_size = (1, kernel // 2 + 1)
    padding = (0, kernel // 2)
    h_stack = ZeroPadding2D(padding=padding)(h_stack_in)
    h_stack = MaskedConv2D(
        kernel_size=kernel_size,
        out_dim=out_dim*2,
        direction="h",
        mode=mask
        )(h_stack)
    #h_stack = h_stack[:, :, :int(h_stack_in.get_shape()[-2]), :]
    h_stack = Lambda(lambda x: x[:, :, :K.int_shape(h_stack_in)[-2], :])(h_stack)
    if use_context:
        h_stack = context_projection(h_stack, context)
    h_stack_1 = Conv2D(
        filters=out_dim*2,
        kernel_size=1,
        strides=(1, 1)
        )(v_stack)
    h_stack_1 = Lambda(lambda x: x[:, :, :K.int_shape(h_stack_in)[-2], :])(h_stack_1)
    h_stack_out = Add()([h_stack, h_stack_1])
    h_stack_out = gated_activation(h_stack_out)
    
    h_stack_out =  Conv2D(
        filters=out_dim,
        kernel_size=1,
        strides=(1, 1)
        )(h_stack_out)
    if residual:
        h_stack_out = Add()([h_stack_in, h_stack_out])
    return v_stack_out, h_stack_out
