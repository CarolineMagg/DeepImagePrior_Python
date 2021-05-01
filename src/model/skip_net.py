####################################################################
# SkipNet implementation
####################################################################

from src.model.model_utils import *

__author__ = "c.magg"


def skip_net(input_shape=(512, 512, 32), batch_size=1, output_size=3,
             nb_filters_down=None, nb_filters_up=None, nb_filters_skip=None,
             need1x1_up=True, need_sigmoid=True,
             downsample_mode="stride", activation="leaky_relu", padding="reflection",
             upsample_mode="bilinear", seed=13317):
    """
    Generate SkipNet
    :param input_shape: network input shape (default: (512,512,32))
    :param batch_size: network input batch size (default: 1)
    :param output_size: network output shape (default: 3)
    :param nb_filters_down: filters in downsampling path (default: [128, 128, 128, 128])
    :param nb_filters_up: filters in upsampling path (default: same as in downsampling)
    :param nb_filters_skip: filters in skip connections (default: [4, 4, 4, 4])
    :param need1x1_up: flag if 1x1 conv is used in upsampling path
    :param need_sigmoid: flag if sigmoid activation is needed at the end
    :param downsample_mode: mode for downsampling (default: "stride" - use conv stride for downsampling)
    :param activation: activation method used in conv blocks (default: "leaky_relu")
    :param padding: padding method used before conv layers
    :param upsample_mode: mode for upsampling (default: "bilinear")
    :param seed: random seeed init
    :return: tf keras model with SkipNet architecture
    """
    tf.random.set_seed(seed)

    if nb_filters_down is None:
        nb_filters_down = [128, 128, 128, 128]
    if nb_filters_up is None:
        nb_filters_up = nb_filters_down
    if len(nb_filters_up) == 1:
        nb_filters_up = [nb_filters_up]*len(nb_filters_down)
    if nb_filters_skip is None:
        nb_filters_skip = [4]*len(nb_filters_down)
    if len(nb_filters_skip) == 1:
        nb_filters_skip = [nb_filters_skip]*len(nb_filters_down)

    depth = len(nb_filters_down) - 1

    input_layer = tf.keras.Input(shape=input_shape, batch_size=batch_size, name='input', dtype=None, sparse=False,
                                 tensor=None)
    x = input_layer

    # downsampling
    skip_block = []
    for idx, nd in enumerate(nb_filters_down):
        # first conv block
        x = convolutional_block(x, filters=nd, kernel_size=3, strides=(2, 2),
                                pad=padding, downsample_mode=downsample_mode,
                                act=activation)
        # second conv block
        x = convolutional_block(x, filters=nd, kernel_size=3, strides=(1, 1),
                                pad=padding, downsample_mode=downsample_mode,
                                act=activation)

        # skip connection
        skip = convolutional_block(x, filters=nb_filters_skip[idx], kernel_size=1, strides=(1, 1),
                                   pad=padding, downsample_mode=downsample_mode,
                                   act=activation)
        skip_block.append(skip)

    # bridge
    x = skip_block.pop(-1)

    # upsampling and concatenate
    for idx, nu in enumerate(reversed(nb_filters_up[1:])):
        # first conv block
        x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-05, name="BN_" + str(idx))(x)
        x = convolutional_block(x, filters=nu, kernel_size=3, strides=(1, 1),
                                pad=padding, downsample_mode=downsample_mode,
                                act=activation)
        # second conv block 1x1 convolution
        if need1x1_up:
            x = convolutional_block(x, filters=nu, kernel_size=1, strides=(1, 1),
                                    pad=padding, downsample_mode=downsample_mode,
                                    act=activation)
        # upsample
        if upsample_mode == "bilinear":
            x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x)
        elif upsample_mode == "nearest":
            x = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(x)
        else:
            raise NotImplementedError("Upsampling {0} not implemented.".format(upsample_mode))

        # concatenate
        x = tf.keras.layers.concatenate([skip_block[depth - 1 - idx], x], axis=3)

    # last step in up filters
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-05)(x)
    x = convolutional_block(x, filters=nb_filters_up[0], kernel_size=3, strides=(1, 1),
                            pad=padding, downsample_mode=downsample_mode,
                            act=activation)
    # second conv block 1x1 convolution
    if need1x1_up:
        x = convolutional_block(x, filters=nb_filters_up[0], kernel_size=1, strides=(1, 1),
                                pad=padding, downsample_mode=downsample_mode,
                                act=activation)

    # last upsample
    if upsample_mode == "bilinear":
        x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    elif upsample_mode == "nearest":
        x = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(x)
    else:
        raise NotImplementedError("Upsampling {0} not implemented.".format(upsample_mode))

    # last conv layer -> filter = 3
    x = convolutional_layer(x, filters=output_size, kernel_size=3, strides=(1, 1),
                            pad=padding, downsample_mode=downsample_mode)
    # sigmoid activation
    if need_sigmoid:
        x = tf.keras.activations.sigmoid(x)

    return tf.keras.Model(inputs=input_layer, outputs=x)
