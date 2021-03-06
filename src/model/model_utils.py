####################################################################
# Utility methods for model inits
# * Reflection Padding layer
# * convolutional layer with padding+conv+downsampling
# * convolutional block with conv layer+bn+act
####################################################################

import tensorflow as tf

__author__ = "c.magg"


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


def convolutional_layer(x, filters, kernel_size, strides,
                        pad='reflection', downsample_mode='stride'):
    # padding
    to_pad = (int((kernel_size - 1) / 2), int((kernel_size - 1) / 2))
    if pad == 'reflection':
        x = ReflectionPadding2D(to_pad)(x)

    # conv2D
    downsample_stride = strides
    if strides[0] != 1 and downsample_mode != 'stride':
        strides = (1, 1)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(x)

    # downsampling
    if downsample_stride[0] != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            x = tf.keras.layers.AveragePooling2D(pool_size=downsample_stride,
                                                 strides=downsample_stride)(x)
        elif downsample_mode == 'max':
            x = tf.keras.layers.MaxPool2D(pool_size=downsample_stride,
                                          strides=downsample_stride)(x)
        else:
            raise ValueError("downsample_mode {0} not valid.".format(downsample_mode))
    return x


def convolutional_block(x, filters, kernel_size, strides,
                        pad='reflection', downsample_mode='stride',
                        act='leaky_relu'):
    # padding + conv + downsampling
    x = convolutional_layer(x, filters, kernel_size, strides, pad, downsample_mode)
    # bn + activation
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-05)(x)
    if act == 'leaky_relu':
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    elif act == 'relu':
        x = tf.keras.activations.relu(x)
    elif act == 'swish':
        x = tf.keras.activations.swish(x)
    elif act == 'mish':
        x = mish(x)
    else:
        raise NotImplementedError("Activation method {0} is not valid.".format(act))
    return x


def convolutional_block_unet(x, filters, kernel_size=3, strides=1, pad="same",
                             norm_layer="batch_norm", activation="relu"):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=pad)(x)
    if norm_layer == "batch_norm":
        x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-05)(x)
    if activation == "relu":
        x = tf.keras.activations.relu(x)
    else:
        raise NotImplementedError("Activation {0} not valid.".format(activation))

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=pad)(x)
    if norm_layer == "batch_norm":
        x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-05)(x)
    if activation == "relu":
        x = tf.keras.activations.relu(x)
    else:
        raise NotImplementedError("Activation {0} not valid.".format(activation))
    return x


def mish(x):
    """
    Mish activation method
    :param x:
    :return:
    """
    return x * tf.math.tanh(tf.math.softplus(x))
