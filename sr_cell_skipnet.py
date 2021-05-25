import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from src.training_pipeline.dip_training_utils import perform_train_run
from src.utils.data_processing import load_image, convert_to_tf_tensor, get_low_resolution_image

import tensorflow as tf


def check_gpu():
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

    checkpoint = 'train_runs/sr/ISBI_2012/'
    seed = 131775
    in_channels = 32

    # load images
    sigma_ = 25 / 255.
    images_path = "data/ISBI_2012/Train-Volume/"
    images = [os.path.join(images_path, fn) for fn in os.listdir(images_path)]

    # loop over all images with standard settings
    for fn in images:
        # load image
        img = load_image(fn)
        img_lr = get_low_resolution_image(img, 4)
        net_img_original = convert_to_tf_tensor(img)
        net_img_noise = convert_to_tf_tensor(img_lr)

        print("training run for {0}".format(fn))

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=os.path.join(checkpoint, fn.replace(images_path, "")),
                          idx=seed, msg="baseline")

    # loop over all images with standard settings
    for fn in images:
        # load image
        img = load_image(fn)
        img_lr = get_low_resolution_image(img, 4)
        net_img_original = convert_to_tf_tensor(img)
        net_img_noise = convert_to_tf_tensor(img_lr)

        print("training run for {0}".format(fn))

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=os.path.join(checkpoint, fn.replace(images_path, "")),
                          act="swish", nb_filters_skip=8,
                          idx=seed, msg="swish_8_skip")

    for fn in images:
        # load image
        img = load_image(fn)
        img_lr = get_low_resolution_image(img, 4)
        net_img_original = convert_to_tf_tensor(img)
        net_img_noise = convert_to_tf_tensor(img_lr)

        print("training run for {0}".format(fn))

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=os.path.join(checkpoint, fn.replace(images_path, "")),
                          act="swish",
                          idx=seed, msg="swish")

    for fn in images:
        # load image
        img = load_image(fn)
        img_lr = get_low_resolution_image(img, 4)
        net_img_original = convert_to_tf_tensor(img)
        net_img_noise = convert_to_tf_tensor(img_lr)

        print("training run for {0}".format(fn))

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=os.path.join(checkpoint, fn.replace(images_path, "")),
                          nb_filters_skip=8,
                          idx=seed, msg="8_skip")

    for fn in images:
        # load image
        img = load_image(fn)
        img_lr = get_low_resolution_image(img, 4)
        net_img_original = convert_to_tf_tensor(img)
        net_img_noise = convert_to_tf_tensor(img_lr)

        print("training run for {0}".format(fn))

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=os.path.join(checkpoint, fn.replace(images_path, "")),
                          nb_filters=[128, 128, 128, 128, 128],
                          idx=seed, msg="5_128_fm")

    for fn in images:
        # load image
        img = load_image(fn)
        img_lr = get_low_resolution_image(img, 4)
        net_img_original = convert_to_tf_tensor(img)
        net_img_noise = convert_to_tf_tensor(img_lr)

        print("training run for {0}".format(fn))

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-08, amsgrad=False, name='Adam')
        perform_train_run(net_img_original, net_img_noise,
                          optimizer=adam, checkpoint_folder=os.path.join(checkpoint, fn.replace(images_path, "")),
                          act="mish",
                          idx=seed, msg="mish")
