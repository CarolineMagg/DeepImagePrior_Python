####################################################################
# Methods for DIP training
# * store results
# * store training history
# * perform training run with CustomModel
####################################################################

import copy
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils.data_processing import get_network_input
from src.model.skip_net import skip_net

__author__ = "c.magg"


def store_history(hist, time_sec, fn):
    """
    Store history to csv file
    :param hist: tf history
    :param time_sec: training time in seconds
    :param fn: filename for csv file
    :return:
    """
    epochs = len(hist.history['loss'])
    df = pd.DataFrame(index=list(hist.history.keys()) + ["Time"], columns=range(epochs))
    for key in hist.history.keys():
        df.loc[key] = hist.history[key]
    df.loc["Time"][0] = time_sec
    df = df.transpose()
    df.to_csv(fn)


def store_result(results, results_corrupted, fn, metric_names):
    """
    Store results of evaluation to csv file
    :param results: results of GT
    :param results_corrupted: results of corrupted image
    :param fn: filename for csv file
    :param metric_names: names of metrics used during training
    :return:
    """
    df = pd.DataFrame(index=metric_names, columns=["GT", "Corrupted"])
    for key, elem in zip(metric_names, results):
        df.loc[key]["GT"] = elem
    for key, elem in zip(metric_names, results_corrupted):
        df.loc[key]["Corrupted"] = elem
    df = df.transpose()
    df.to_csv(fn)


def perform_train_run(img, img_corrupted,
                      optimizer, checkpoint_folder, idx,
                      act="leaky_relu", up="bilinear", down="stride",
                      in_channels=32, nb_filters=None, nb_filters_skip=4,
                      num_iter=3000, early_stopping_flag=False,
                      msg=""):
    """
    Perform a trainings run with
    * init net input
    * init model, compile model
    * setup callbacks
    * perform training
    * perform evaluation
    * store results and trainings history
    :param idx: index to be used for training run (eg seed used)
    :param img: original image in netput shape (1,H,W,C)
    :param img_corrupted: corrupted image in netput shape (1,H,W,C)
    :param optimizer: optimizer
    :param checkpoint_folder: name of checkpoint folder
    :param act: activation function (default: "leaky_relu")
    :param up: upsampling mode (default: "bilinear")
    :param down: downsampling mode (default: "stride")
    :param in_channels: input channels (default: 32)
    :param nb_filters: number of feature maps in encoder part (default: 4x128)
    :param nb_filters_skip: number of feature maps in skip connections (default: 4)
    :param num_iter: number of iterations (default: 3000)
    :param early_stopping_flag: flag for early stopping (default: false)
    :param msg:
    :return:
    """
    net_img = copy.deepcopy(img)
    net_img_corrupted = copy.deepcopy(img_corrupted)

    if nb_filters is None:
        nb_filters = [128, 128, 128, 128]

    # net input
    input_shape = (img.shape[0], img.shape[1], img.shape[2], in_channels)
    net_input = get_network_input(input_shape)

    # checkpoint
    print("\nTraining run {0} with {1}".format(idx, msg))
    checkpoint_name = os.path.join(checkpoint_folder, msg, str(idx))
    print("Checkpoint name {0}".format(checkpoint_name))
    if not os.path.isdir(checkpoint_name):
        print("Checkpoint name does not exist. Create checkpoint name.")
        os.makedirs(checkpoint_name)

    # init model
    model = skip_net(input_shape=input_shape[1:], activation=act, downsample_mode=down, upsample_mode=up,
                     nb_filters_down=nb_filters, nb_filters_skip=nb_filters_skip)
    model.compile(optimizer=optimizer)

    # callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_psnr', patience=100, verbose=1, mode="max")
    fn = os.path.join(checkpoint_name, "model_weights.hdf5")
    cbs = [tf.keras.callbacks.ModelCheckpoint(
        filepath=fn,
        save_weights_only=True,
        monitor='val_psnr',
        mode='max',
        save_best_only=True,
        verbose=0)]
    if early_stopping_flag:
        cbs.append(early_stopping)

    # training
    start = time.time()
    history = model.fit(x=net_input,
                        y=net_img_corrupted,
                        validation_data=(net_input, net_img),
                        epochs=num_iter,
                        verbose=2,
                        callbacks=cbs)

    # evaluation
    time_elapsed = time.time() - start
    print("Training finished after {0} sec.".format(np.round(time_elapsed, decimals=3)))
    store_history(history, time_elapsed, os.path.join(checkpoint_name, "history.csv"))
    model.load_weights(fn)
    res = model.evaluate(x=net_input,
                         y=net_img)
    res2 = model.evaluate(x=net_input,
                          y=net_img_corrupted)
    store_result(res, res2, os.path.join(checkpoint_name, "results.csv"), model.metrics_names)
