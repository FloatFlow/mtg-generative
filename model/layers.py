# imports
import numpy as np
from keras.layers import Layer, Dense, Conv2D, Conv2DTranspose, Embedding, InputSpec, Conv1D, \
                         Multiply, Add, Conv3D, Reshape, UpSampling2D
import keras.backend as K
#from keras.models import 
import tensorflow as tf
from keras import initializers
from keras.initializers import VarianceScaling
from keras.utils import conv_utils
from keras import regularizers, constraints
import keras

from tensorflow.python.training import moving_averages

class MaskedConv2D(Layer):
    """
    Masked convolution
    Adapted from https://www.kaggle.com/ameroyer/keras-vq-vae-for-image-generation
    """
    def __init__(self, kernel_size, out_dim, direction, mode, **kwargs):
        self.direction = direction     # Horizontal or vertical
        self.mode = mode               # Mask type "a" or "b"
        self.kernel_size = kernel_size
        self.out_dim = out_dim
        super(MaskedConv2D, self).__init__(**kwargs)
    
    def build(self, input_shape):   
        filter_mid_y = self.kernel_size[0] // 2
        filter_mid_x = self.kernel_size[1] // 2        
        in_dim = int(input_shape[-1])
        w_shape = [self.kernel_size[0], self.kernel_size[1], in_dim, self.out_dim]
        mask_filter = np.ones(w_shape, dtype=np.float32)
        # Build the mask
        if self.direction == "h":
            mask_filter[filter_mid_y + 1:, :, :, :] = 0.
            mask_filter[filter_mid_y, filter_mid_x + 1:, :, :] = 0.
        elif self.direction == "v":
            if self.mode == 'a':
                mask_filter[filter_mid_y:, :, :, :] = 0.
            elif self.mode == 'b':
                mask_filter[filter_mid_y+1:, :, :, :] = 0.0
        if self.mode == 'a':
            mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.0
        # Create convolution layer parameters with masked kernel
        mask_filter = K.constant(mask_filter)
        self.W = mask_filter * self.add_weight(
            name="W_{}".format(self.direction),
            shape=w_shape,
            trainable=True,
            initializer='glorot_uniform'
            )
        self.b = self.add_weight(
            name="v_b",
            shape=[self.out_dim, ],
            trainable=True,
            initializer='zeros'
            )
        super(MaskedConv2D, self).build(input_shape)
    
    def call(self, inputs):
        x = K.conv2d(inputs, self.W, strides=(1, 1))
        x = K.bias_add(x, self.b)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.out_dim)

class VectorQuantizer(Layer):  
    def __init__(self, k, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.k = k
    
    def build(self, input_shape):
        self.d = int(input_shape[-1])
        self.codebook = self.add_weight(
            shape=(self.k, self.d),
            initializer=VarianceScaling(distribution="uniform"),
            trainable=True,
            name='codebook'
            )
        
    def call(self, inputs):
        # Map z_e of shape (b, w,, h, d) to indices in the codebook
        lookup_ = K.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        z_e = K.expand_dims(inputs, -2)
        dist = tf.norm(z_e - lookup_, axis=-1)
        k_index = K.argmin(dist, axis=-1)
        return k_index
    
    def sample(self, k_index):
        # Map indices array of shape (b, w, h) to actual codebook z_q
        lookup_ = K.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        k_index_one_hot = K.one_hot(k_index, self.k)
        z_q = lookup_ * k_index_one_hot[..., None]
        z_q = tf.reduce_sum(z_q, axis=-2)
        return z_q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])

class ReparameterizationTrick(Layer):
    def __init__(self, **kwargs):
        super(ReparameterizationTrick, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReparameterizationTrick, self).build(input_shape)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class ModulatedConv2D(Layer):
    def __init__(
        self,
        kernel_size=3,
        filters=256,
        padding='same',
        upsample=False,
        downsample=False,
        kernel_initializer='he_uniform',
        demodulate=True,
        **kwargs
        ):
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise ValueError('Invalid kernel size dtype for ModulatedConv2D')
        self.filters = filters
        self.padding = padding
        self.upsample = upsample
        self.downsample = downsample
        self.kernel_initializer = kernel_initializer
        self.demodulate = demodulate
        super(ModulatedConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        img_shape, style_shape = input_shape
        input_dim = img_shape[-1]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            )
        #super(ModulatedConv2D, self).build(input_shape)
    
    def call(self, inputs):
        input_vals, style = inputs
        # add minibatch dim
        mod_w = K.expand_dims(self.kernel, axis=0) #(BkkIO) 

        # modulate
        mod_w = mod_w * K.reshape(style, (-1, 1, 1, K.int_shape(style)[-1], 1)) #(BkkIO)

        if self.demodulate:
            # demodulate
            mod_d = tf.math.rsqrt(tf.reduce_sum(tf.square(mod_w), axis=[1,2,3]) + K.epsilon())

        # scale input activations
        x = input_vals * K.reshape(style, (-1, 1, 1, K.int_shape(style)[-1])) #(BhwI)

        if self.upsample:
            x = UpSampling2D(2, interpolation='bilinear')(x)
            x = K.conv2d(
                x,
                self.kernel,
                padding=self.padding
                )
        elif self.downsample:
            x = K.conv2d(
                x,
                self.kernel,
                strides=(2, 2),
                padding=self.padding
                )
        else:
            x = K.conv2d(
                x,
                self.kernel,
                padding=self.padding
                )
        if self.demodulate:
            # scale output activations
            x = x * K.reshape(mod_d, (-1, 1, 1, K.int_shape(mod_d)[-1])) #(BhwO)
        return x

    def compute_output_shape(self, input_shape):
        img_shape, style_shape = input_shape
        if self.downsample:
            img_shape = (img_shape[0], img_shape[1]//2, img_shape[2]//2, self.filters)
        elif self.upsample:
            img_shape = (img_shape[0], img_shape[1]*2, img_shape[2]*2, self.filters)
        else:
            img_shape = (img_shape[0], img_shape[1], img_shape[2], self.filters)
        return img_shape

class ModulatedSNConv2D(Layer):
    def __init__(
        self,
        kernel_size=3,
        filters=256,
        padding='same',
        upsample=False,
        downsample=False,
        kernel_initializer='he_uniform',
        demodulate=True,
        **kwargs
        ):
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise ValueError('Invalid kernel size dtype for ModulatedConv2D')
        self.filters = filters
        self.padding = padding
        self.upsample = upsample
        self.downsample = downsample
        self.kernel_initializer = kernel_initializer
        self.demodulate = demodulate
        super(ModulatedSNConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        img_shape, style_shape = input_shape
        input_dim = img_shape[-1]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            )

        self.u = self.add_weight(
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=initializers.RandomNormal(0, 1),
            name='sn',
            trainable=False
            )
        #super(ModulatedConv2D, self).build(input_shape)
    
    def call(self, inputs, training=None):
        input_vals, style = inputs
        ### spectral normalization ###
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)

        ### modulation/demodulation ###
        # add minibatch dim
        mod_w = K.expand_dims(W_bar, axis=0) #(BkkIO)

        # modulate
        mod_w = mod_w * K.reshape(style, (-1, 1, 1, K.int_shape(style)[-1], 1)) #(BkkIO)

        if self.demodulate:
            # demodulate
            mod_d = tf.math.rsqrt(tf.reduce_sum(tf.square(mod_w), axis=[1,2,3]) + K.epsilon())

        # scale input activations
        x = input_vals * K.reshape(style, (-1, 1, 1, K.int_shape(style)[-1])) #(BhwI)

        if self.upsample:
            x = UpSampling2D(2, interpolation='bilinear')(x)
            x = K.conv2d(
                x,
                W_bar,
                padding=self.padding
                )
        elif self.downsample:
            x = K.conv2d(
                x,
                W_bar,
                strides=(2, 2),
                padding=self.padding
                )
        else:
            x = K.conv2d(
                x,
                W_bar,
                padding=self.padding
                )
        if self.demodulate:
            # scale output activations
            x = x * K.reshape(mod_d, (-1, 1, 1, K.int_shape(mod_d)[-1])) #(BhwO)
        return x

    def compute_output_shape(self, input_shape):
        img_shape, style_shape = input_shape
        if self.downsample:
            img_shape = (img_shape[0], img_shape[1]//2, img_shape[2]//2, self.filters)
        elif self.upsample:
            img_shape = (img_shape[0], img_shape[1]*2, img_shape[2]*2, self.filters)
        else:
            img_shape = (img_shape[0], img_shape[1], img_shape[2], self.filters)
        return img_shape

class Bias(Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(
            name='bias',
            shape=(input_shape[-1], ),
            initializer='zeros',
            trainable=True
            )
        super(Bias, self).build(input_shape)

    def call(self, x):
        return x + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

class NoiseLayer(Layer):
    def __init__(self, **kwargs):
        super(NoiseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.noise_weight = self.add_weight(
            'noise_weight',
            shape=[1, ],
            initializer='zeros'
            )

    def call(self, x, **kwargs):
        noise = K.random_normal([1, K.shape(x)[1], K.shape(x)[2], 1], dtype=x.dtype)  # [batch, h, w, c]
        return x + noise * self.noise_weight

    def compute_output_shape(self, input_shape):
        return input_shape


class LowPassFilter2D(Layer):
    def __init__(self, kernel_size=3, **kwargs):
        self.kernel_size = kernel_size
        super(LowPassFilter2D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.kernel_size == 2:
            self.kernel = np.array([[1, 1],
                                    [1, 1]])
        else:
            self.kernel = np.array([[1, 1, 1],
                                    [1, 2, 1],
                                    [1, 1, 1]])
        self.kernel = self.kernel/np.sum(self.kernel)
        self.kernel = K.constant(self.kernel)
        super(LowPassFilter2D, self).build(input_shape)
    
    def blur2d(self, inputs, flip=False):
        n_channels = K.int_shape(inputs)[-1]
        kernel = K.expand_dims(self.kernel, axis=-1)
        kernel = K.expand_dims(kernel, axis=-1)
        kernel = K.tile(kernel, (1, 1, n_channels, 1))
        convolved = K.depthwise_conv2d(
            inputs,
            kernel,
            strides=(1, 1),
            padding='same',
            data_format='channels_last',
            dilation_rate=(1, 1)
            )
        if flip:
            convolved = convolved[:, :, ::-1, :]
        return convolved

    def call(self, inputs):
        @tf.custom_gradient
        def func(x):
            y = self.blur2d(x)
            @tf.custom_gradient
            def grad(dy):
                dx = self.blur2d(dy, flip=True)
                return dx, lambda ddx: self.blur2d(ddx)
            return y, grad
        return func(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

class LearnedConstantLatent(Layer):
    def __init__(self, latent_size=None, **kwargs):
        self.latent_size = latent_size
        super(LearnedConstantLatent, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.latent_size == None:
            self.latent_size = input_shape[-1]

        self.latent_weights = self.add_weight(
            shape=(4, 4, self.latent_size),
            name='learned_latent',
            initializer='ones'
            )
        super(LearnedConstantLatent, self).build(input_shape)

    def call(self, inputs):
        batch_size = K.shape(inputs)[0]
        latent_expand = K.expand_dims(self.latent_weights, axis=0)
        latent_expand = K.tile(latent_expand, (batch_size, 1, 1, 1))
        return latent_expand

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4, 4, self.latent_size)

class LatentPixelNormalization(Layer):
    def __init__(self, **kwargs):

        super(LatentPixelNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LatentPixelNormalization, self).build(input_shape)

    def call(self, inputs):
        latent = inputs * (K.constant(1.0)/ K.sqrt(K.mean(K.square(inputs), axis=-1, keepdims=True) + K.epsilon()))
        return latent

    def compute_output_shape(self, input_shape):
        return input_shape

class ConstantLatent(Layer):
    def __init__(self, **kwargs):
        super(ConstantLatent, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ConstantLatent, self).build(input_shape)

    def call(self, inputs):
        input_shape = K.shape(inputs)
        latent = K.ones((input_shape[0], 4, 4, input_shape[-1]))
        return latent

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4, 4, input_shape[-1])

class MiniBatchStd(Layer):
    def __init__(self, group_size=4, n_new_features=1, **kwargs):
        self.group_size = group_size
        self.n_new_features = n_new_features
        super(MiniBatchStd, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MiniBatchStd, self).build(input_shape)

    def call(self, inputs):
        
        group_size = K.minimum(self.group_size, K.shape(inputs)[0])
        y = K.permute_dimensions(inputs, (0, 3, 1, 2))
        input_shape = K.shape(y)
        y = K.reshape(y, [group_size,
                          -1,
                          self.n_new_features,
                          input_shape[1]//self.n_new_features,
                          input_shape[2],
                          input_shape[3]])
        y -= K.mean(y, axis=0, keepdims=True)
        y = K.mean(K.square(y), axis=0)
        y = K.sqrt(y + K.epsilon())
        y = K.mean(y, axis=[2, 3, 4], keepdims=True)
        y = K.mean(y, axis=2)
        y = K.tile(y, [group_size, 1, input_shape[2], input_shape[3]])
        y = K.permute_dimensions(y, (0, 2, 3, 1))
        
        #y = K.std(inputs, axis=-1, keepdims=True)
        outputs = K.concatenate([inputs, y], axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3]+self.n_new_features)

#Input b and g should be 1x1xC
class InstanceNormalization(Layer):
    def __init__(self, 
             axis=-1,
             momentum=0.99,
             epsilon=1e-3,
             center=True,
             scale=True,
             **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
    
    
    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        self.gamma = self.add_weight(
            shape=shape,
            name='gamma',
            initializer='ones'
            )

        self.beta = self.add_weight(
            shape=shape,
            name='beta',
            initializer='zeros'
            )

        self.built = True
        super(InstanceNormalization, self).build(input_shape) 
    
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape

#Input b and g should be 1x1xC
class AdaInstanceNormalization(Layer):
    def __init__(self, 
             axis=-1,
             momentum=0.99,
             epsilon=1e-3,
             center=True,
             scale=True,
             **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
    
    
    def build(self, input_shape):
        
        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')
    
        super(AdaInstanceNormalization, self).build(input_shape) 
    
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        reduction_axes = list(range(1, len(input_shape)))
        
        beta = inputs[1]
        gamma = inputs[2]

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
    
        return input_shape[0]

'''
class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
'''

class GlobalSumPooling2D(Layer):
    def __init(self, **kwargs):
        super(GlobalSumPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GlobalSumPooling2D, self).build(input_shape)
        # Set input spec.
        #self.input_spec = InputSpec(ndim=4,
        #                            axes={3: input_shape[-1]})
        self.built = True

    def call(self, inputs):
        gsp = tf.reduce_sum(inputs, axis=[1, 2])
        return gsp

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class PixelNormalization(Layer):
    def __init(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PixelNormalization, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True

    def call(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape

class LayerNormalization(Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = keras.engine.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

##########################
## Attention Layers
##########################
class AttentionBlock(Layer):
    def __init__(self, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttentionBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # check that the implementation matches exactly py torch.
        keys, queries, values, original = inputs

        logits = K.batch_dot(queries, K.permute_dimensions(keys, (0, 2, 1)))
        mask = K.ones_like(logits) * np.triu((-np.inf) * np.ones(logits.shape.as_list()[1:]), k=1)
        logits = mask + logits
        probs = K.softmax(logits, axis=-1)
        read = K.batch_dot(probs, values)
        output = K.concatenate([original, read], axis=-1)
        print("Warning - attention output: {}".format(K.int_shape(output)))
        return output
    def compute_output_shape(self, input_shape):
        k_shape, q_shape, v_shape, _ = input_shape
        return (k_shape[0], k_shape[1], k_shape[2]+q_shape[2]+v_shape[2])

"""
class AttentionBlock(Layer):
    def __init__(self, dims, k_size, v_size, seq_len=None, **kwargs):
        self.k_size = k_size
        self.seq_len = seq_len
        self.v_size = v_size
        self.dims = dims
        self.sqrt_k = np.sqrt(k_size)
        self.keys_fc = None
        self.queries_fc = None
        self.values_fc = None
        super(AttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        # https://stackoverflow.com/questions/54194724/how-to-use-keras-layers-in-custom-keras-layer
        #self.keys_fc = Dense(self.k_size)
        #self.keys_fc.build((None, self.dims))
        #self._trainable_weights.extend(self.keys_fc.trainable_weights)
#
        #self.queries_fc = Dense(self.k_size)
        #self.queries_fc.build((None, self.dims))
        #self._trainable_weights.extend(self.queries_fc.trainable_weights)
#
        #self.values_fc = Dense(self.v_size)
        #self.values_fc.build((None, self.dims))
        #self._trainable_weights.extend(self.values_fc.trainable_weights)
        super(AttentionBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # check that the implementation matches exactly py torch.
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[-1]])

        original_shape = K.shape(inputs)
        inputs = hw_flatten(inputs)

        keys = self.keys_fc(inputs)
        queries = self.queries_fc(inputs)
        values = self.values_fc(inputs)

        logits = K.batch_dot(queries, K.permute_dimensions(keys, (0, 2, 1)))
        mask = K.ones_like(logits) * np.triu((-np.inf) * np.ones(logits.shape.as_list()[1:]), k=1)
        logits = mask + logits
        probs = Softmax(axis=-1)(logits / self.sqrt_k)
        read = K.batch_dot(probs, values)
        output = K.concatenate([inputs, read], axis=-1)
        #output = K.reshape(output, (original_shape[0], original_shape[1], original_shape[2], -1))
        return output

    #def compute_output_shape(self, input_shape):
    #    output_shape = list(input_shape)
    #    output_shape[-1] += self.v_size
    #    return tuple(output_shape)
"""
class ScaledDotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        filter_shape = (self.channels, self.channels)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_v = self.add_weight(
            shape=filter_shape,
            initializer='glorot_uniform',
            name='kernel_v')
        self.kernel_k = self.add_weight(
            shape=filter_shape,
            initializer='glorot_uniform',
            name='kernel_k')
        self.kernel_q = self.add_weight(
            shape=filter_shape,
            initializer='glorot_uniform',
            name='kernel_q')

        super(ScaledDotProductAttention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True


    def call(self, inputs):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[-1]])

        v = K.dot(inputs, self.kernel_v)
        k = K.dot(inputs, self.kernel_k)
        q = K.dot(inputs, self.kernel_q)

        s = tf.matmul(hw_flatten(v), hw_flatten(k), transpose_b=True)  # # [bs, N, N]
        s = s*self.gamma

        # masking
        #mask = K.ones_like(s) * np.triu((-np.inf) * np.ones(s.shape.as_list()[1:]), k=1)
        #s = mask + s

        beta = K.softmax(s, axis=-1)  # attention map
        o = tf.matmul(beta, hw_flatten(q))
        outputs = K.reshape(o, shape=K.shape(inputs))  # [bs, h, w, C]
        return outputs+inputs

    def compute_output_shape(self, input_shape):
        return input_shape

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')

        # build masks

        super(Attention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True



    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[-1]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = K.bias_add(f, self.bias_f)
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.bias_add(g, self.bias_g)
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = K.bias_add(h, self.bias_h)

        #s = K.batch_dot(hw_flatten(g), K.transpose(hw_flatten(f)))

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]


        beta = K.softmax(s, axis=-1)  # attention map

        #o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]
        o = tf.matmul(beta, hw_flatten(h))

        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

# google version of attention
class Attention2(Layer):
    def __init__(self, ch, **kwargs):
        super(Attention2, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels // 2

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)
        kernel_shape_o = (1, 1) + (self.channels // 2, self.channels)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.kernel_o = self.add_weight(shape=kernel_shape_o,
                                        initializer='glorot_uniform',
                                        name='kernel_o')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')
        super(Attention2, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True


    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[-1]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = K.bias_add(f, self.bias_f)
        f = K.pool2d(f, (2,2), strides=(2, 2), padding='same', pool_mode='max')

        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.bias_add(g, self.bias_g)
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = K.bias_add(h, self.bias_h)
        h = K.pool2d(h, (2,2), strides=(2, 2), padding='same', pool_mode='max')

        #s = K.batch_dot(hw_flatten(g), keras.backend.transpose(hw_flatten(f)))
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = K.softmax(s, axis=-1)  # attention map

        #o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]
        o = tf.matmul(beta, hw_flatten(h))

        #o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        #o = tf.reshape(o, shape=[x.shape[0], x.shape[1], x.shape[2], channels // 2])  # [bs, h, w, C]
        o = K.reshape(o, shape=[K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], self.channels // 2])
        o = K.conv2d(o,
                     kernel=self.kernel_o,
                     strides=(1, 1),
                     padding='same')
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

# google version of attention with SN
class Attention2SN(Layer):
    def __init__(self, ch, **kwargs):
        super(Attention2SN, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels // 2

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        #print(kernel_shape_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)
        kernel_shape_o = (1, 1) + (self.channels // 2, self.channels)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.kernel_o = self.add_weight(shape=kernel_shape_o,
                                        initializer='glorot_uniform',
                                        name='kernel_o')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_f')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')

        # add spectral normalization weights
        self.u_f = self.add_weight(shape=tuple([1, self.kernel_f.shape.as_list()[-1]]),
                                   initializer=initializers.RandomNormal(0, 1),
                                   name='sn_f',
                                   trainable=False)

        self.u_g = self.add_weight(shape=tuple([1, self.kernel_g.shape.as_list()[-1]]),
                                   initializer=initializers.RandomNormal(0, 1),
                                   name='sn_g',
                                   trainable=False)

        self.u_h = self.add_weight(shape=tuple([1, self.kernel_h.shape.as_list()[-1]]),
                                   initializer=initializers.RandomNormal(0, 1),
                                   name='sn_h',
                                   trainable=False)
        self.u_o = self.add_weight(shape=tuple([1, self.kernel_o.shape.as_list()[-1]]),
                                   initializer=initializers.RandomNormal(0, 1),
                                   name='sn_o',
                                   trainable=False)

        super(Attention2SN, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True


    def call(self, x):
        # our helper functions
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[-1]])
        

        
        ## f SN and conv
        #Spectral Normalization
        W_shape = self.kernel_f.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel_f, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u_f)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        with tf.control_dependencies([self.u_f.assign(_u)]):
            W_bar = K.reshape(W_bar, W_shape) 
        f = K.conv2d(x,
                     kernel=W_bar,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = K.bias_add(f, self.bias_f)
        f = K.pool2d(f, (2,2), strides=(2, 2), padding='same', pool_mode='max')



        ## g SN and conv
        #Spectral Normalization
        W_shape = self.kernel_g.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel_g, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u_g)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        with tf.control_dependencies([self.u_g.assign(_u)]):
            W_bar = K.reshape(W_bar, W_shape) 
        g = K.conv2d(x,
                     kernel=W_bar,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.bias_add(g, self.bias_g)



        ## h SN and conv
        #Spectral Normalization
        W_shape = self.kernel_h.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel_h, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u_h)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        with tf.control_dependencies([self.u_h.assign(_u)]):
            W_bar = K.reshape(W_bar, W_shape) 
        h = K.conv2d(x,
                     kernel=W_bar,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = K.bias_add(h, self.bias_h)
        h = K.pool2d(h, (2,2), strides=(2, 2), padding='same', pool_mode='max')


        ## Generate attention map
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = K.softmax(s, axis=-1)  # attention map
        o = tf.matmul(beta, hw_flatten(h))

        o = K.reshape(o, shape=[K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], self.channels // 2])

        ## SN for kernel o
        #Spectral Normalization
        W_shape = self.kernel_o.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel_o, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u_o)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        with tf.control_dependencies([self.u_o.assign(_u)]):
            W_bar = K.reshape(W_bar, W_shape) 
        
        o = K.conv2d(o,
                     kernel=W_bar,
                     strides=(1, 1),
                     padding='same')
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

###############
## SN Layers
###############

class DenseSN(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                 W_bar = K.reshape(W_bar, W_shape)  
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output 
        
class _ConvSN(Layer):

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 spectral_normalization=True,
                 **kwargs):
        super(_ConvSN, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.spectral_normalization = spectral_normalization
        self.u = None
        
    def _l2normalize(self, v, eps=1e-12):
        return v / (K.sum(v ** 2) ** 0.5 + eps)
    
    def power_iteration(self, u, W):
        '''
        Accroding the paper, we only need to do power iteration one time.
        '''
        v = self._l2normalize(K.dot(u, K.transpose(W)))
        u = self._l2normalize(K.dot(v, W))
        return u, v
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        #Spectral Normalization
        if self.spectral_normalization:
            self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                     initializer=initializers.RandomNormal(0, 1),
                                     name='sn',
                                     trainable=False)
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        
        if self.spectral_normalization:
            W_shape = self.kernel.shape.as_list()
            #Flatten the Tensor
            W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
            _u, _v = power_iteration(W_reshaped, self.u)
            #Calculate Sigma
            sigma=K.dot(_v, W_reshaped)
            sigma=K.dot(sigma, K.transpose(_u))
            #normalize it
            W_bar = W_reshaped / sigma
            #reshape weight tensor
            if training in {0, False}:
                W_bar = K.reshape(W_bar, W_shape)
            else:
                with tf.control_dependencies([self.u.assign(_u)]):
                    W_bar = K.reshape(W_bar, W_shape)

            #update weitht
            self.kernel = W_bar
        
        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                self.kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(_Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class ConvSN2D(Conv2D):

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)
        
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
                
        outputs = K.conv2d(
                inputs,
                W_bar,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    
class ConvSN1D(Conv1D):
    
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        self.u = self.add_weight(
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                 initializer=initializers.RandomNormal(0, 1),
                 name='sn',
                 trainable=False)
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
                
        outputs = K.conv1d(
                inputs,
                W_bar,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class ConvSN3D(Conv3D):    
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
                
        outputs = K.conv3d(
                inputs,
                W_bar,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

        
class EmbeddingSN(Embedding):
    
    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)
        
        self.u = self.add_weight(shape=tuple([1, self.embeddings.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)
        
        self.built = True
        
    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
            
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.embeddings.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.embeddings, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
        self.embeddings = W_bar
            
        out = K.gather(self.embeddings, inputs)
        return out 

class ConvSN2DTranspose(Conv2DTranspose):

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)
        
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True  
    
    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_length(height,
                                              stride_h, kernel_h,
                                              self.padding,
                                              out_pad_h)
        out_width = conv_utils.deconv_length(width,
                                             stride_w, kernel_w,
                                             self.padding,
                                             out_pad_w)
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)
            
        #Spectral Normalization    
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        #if training in {0, False}:
        #    W_bar = K.reshape(W_bar, W_shape)
        #else:
        with tf.control_dependencies([self.u.assign(_u)]):
            W_bar = K.reshape(W_bar, W_shape)
        self.kernel = W_bar
        
        outputs = K.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape,
            self.strides,
            padding=self.padding,
            data_format=self.data_format)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class MinibatchDiscrimination(Layer):
    """Concatenates to each sample information about how different the input
    features for that sample are from features of other samples in the same
    minibatch, as described in Salimans et. al. (2016). Useful for preventing
    GANs from collapsing to a single output. When using this layer, generated
    samples and reference samples should be in separate batches.
    # Example
    ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)
        # flatten the output so it can be fed into a minibatch discrimination layer
        model.add(Flatten())
        # now model.output_shape == (None, 640)
        # add the minibatch discrimination layer
        model.add(MinibatchDiscrimination(5, 3))
        # now model.output_shape = (None, 645)
    ```
    # Arguments
        nb_kernels: Number of discrimination kernels to use
            (dimensionality concatenated to output).
        kernel_dim: The dimensionality of the space where closeness of samples
            is calculated.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(samples, input_dim + nb_kernels)`.
    # References
        - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
    """

    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
            initializer=self.init,
            name='kernel',
            regularizer=self.W_regularizer,
            trainable=True,
            constraint=self.W_constraint)

        # Set built to true.
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1]+self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
