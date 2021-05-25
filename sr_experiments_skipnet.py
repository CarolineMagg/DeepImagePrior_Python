import copy
import os
import time
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from src.training_pipeline.dip_training_utils import perform_train_run
from src.utils.data_processing import load_image, convert_to_tf_tensor, get_low_resolution_image

import tensorflow as tf


def check_gpu():
    """
    Setup GPU and tensorflow session
    :return:
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


if __name__ == "__main__":
    check_gpu()

    # standard checkpoint folder
    checkpoint = 'train_runs/sr/original/'
    seeds = [131775, 356684, 156689]
    num_iter = 2000

    # load images
    sigma_ = 25 / 255.
    img = load_image("data/original/zebra_GT.png")
    img_lr = get_low_resolution_image(img, 4)
    net_img_original = convert_to_tf_tensor(img)
    net_img_noise = convert_to_tf_tensor(img_lr)

    # perform runs 3 times to get an average
    for seed in seeds:
        # baseline
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          msg="baseline", num_iter=num_iter)

    for seed in seeds:
        # activation function
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          act="relu", msg="relu", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          act="mish", msg="mish", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          act="swish", msg="swish", num_iter=num_iter)

        # down- and upsampling
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          down="max", msg="max", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          down="avg", msg="avg", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          up="nearest", msg="nearest", num_iter=num_iter)

    for seed in seeds:
        # input channels
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          in_channels=1, msg="1_channel", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          in_channels=3, msg="3_channel", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          in_channels=16, msg="16_channel", num_iter=num_iter)

    for seed in seeds:
        # feature maps
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          nb_filters=[128, 128, 128], msg="3_128_fm", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          nb_filters=[128, 128, 128, 128, 128], msg="5_128_fm", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          nb_filters=[64, 64, 64, 64], msg="4_64_fm", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          nb_filters=[32, 32, 32, 32], msg="4_32_fm", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          nb_filters=[16, 32, 64, 128], msg="16_32_64_128", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          nb_filters=[16, 32, 64], msg="16_32_64", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          nb_filters=[32, 64, 128], msg="32_64_128", num_iter=num_iter)

        # skip connection
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          nb_filters_skip=8, msg="8_skip_connection", num_iter=num_iter)

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          nb_filters_skip=1, msg="1_skip_connection", num_iter=num_iter)

    for seed in seeds:
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=checkpoint, idx=seed,
                          nb_filters_skip=8, act="swish",
                          msg="swish_8_skip", num_iter=num_iter)
