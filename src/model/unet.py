####################################################################
# UNet implementation
####################################################################

from src.model.model_utils import *

__author__ = "c.magg"


def unet(input_shape=(512, 512, 32), batch_size=1, output_size=3,
         nb_filters_down=None, nb_filters_up=None, nb_filters_skip=None,
         need_sigmoid=True, activation="relu", upsample_mode="deconv",
         drop_out=None, seed=13317):
    """
    Generate Unet Variation
    :param input_shape: network input shape (default: (512,512,32))
    :param batch_size: network input batch size (default: 1)
    :param output_size: network output shape (default: 3)
    :param nb_filters_down: filters in downsampling path (default: [128, 128, 128, 128])
    :param nb_filters_up: filters in upsampling path (default: same as in downsampling)
    :param nb_filters_skip: filters in skip connections (default: [4, 4, 4, 4])
    :param need_sigmoid: flag if sigmoid activation is needed at the end
    :param activation: activation method used in conv blocks (default: "leaky_relu")
    :param upsample_mode: mode for upsampling (default: "bilinear")
    :param drop_out: rate of drop out, if None: no dropout (default: None)
    :param seed: random seeed init
    :return: tf keras model with SkipNet architecture
    """

    tf.random.set_seed(seed)

    if nb_filters_down is None:
        nb_filters_down = [128, 128, 128, 128]
    if nb_filters_up is None:
        nb_filters_up = nb_filters_down
    if len(nb_filters_up) == 1:
        nb_filters_up = [nb_filters_up] * len(nb_filters_down)
    if nb_filters_skip is None:
        nb_filters_skip = [4] * len(nb_filters_down)
    if len(nb_filters_skip) == 1:
        nb_filters_skip = [nb_filters_skip] * len(nb_filters_down)

    depth = len(nb_filters_down) - 1

    input_layer = tf.keras.Input(shape=input_shape, batch_size=batch_size, name='input', dtype=None, sparse=False,
                                 tensor=None)
    x = input_layer

    # downsampling
    down_block = []
    for idx, nd in enumerate(nb_filters_down[:-1]):
        # double Conv2D block
        x = convolutional_block_unet(x, filters=nd, activation=activation, pad="same")
        down_block.append(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        if drop_out is not None:
            x = tf.keras.layers.Dropout(rate=drop_out)(x)

    # bridge
    x = convolutional_block_unet(x, filters=nb_filters_down[-1], activation=activation, pad="same")

    # upsampling and concatenate
    for idx, nu in enumerate(reversed(nb_filters_up[:-1])):

        # upsampling
        if upsample_mode == "deconv":
            x = tf.keras.layers.Conv2DTranspose(nu, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
        elif upsample_mode == "bilinear" or upsample_mode == "nearest":
            x = tf.keras.layers.UpSampling2D((2, 2), interpolation=upsample_mode)(x)
            x = tf.keras.layers.Conv2D(filters=nu, kernel_size=(3, 3))

        # concatenate
        x = tf.keras.layers.concatenate([down_block[depth - 1 - idx], x], axis=3)

        if drop_out is not None:
            x = tf.keras.layers.Dropout(rate=drop_out)(x)

        # double Conv2D block without normalization
        x = convolutional_block_unet(x, filters=nu, activation=activation, norm_layer=None, pad="same")

    # last conv layer -> filter = 3
    x = tf.keras.layers.Conv2D(filters=output_size, kernel_size=3, strides=1, padding="same")(x)
    # sigmoid activation
    if need_sigmoid:
        x = tf.keras.activations.sigmoid(x)

    return tf.keras.Model(inputs=input_layer, outputs=x)
