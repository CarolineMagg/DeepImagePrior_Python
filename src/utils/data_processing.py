####################################################################
# Processing methods for data handling:
# * load image from path
# * generate low resolution image
# * generate noisy image
# * upsampling method to generate baseline images
# * convert tf tensor to np array and vice versa
# * create cut outs
####################################################################

import cv2
import numpy as np
import os
import tensorflow as tf

__author__ = "c.magg"


def load_image(fn, shape=None, crop_div32=True):
    """
    Load image and process it
    :param fn: file name of image
    :param shape: output image shape (default: None) - if None, no resize will be performed)
    :param crop_div32: boolean to crop center to have image size dividable by 32
    return: image with RGB, range 0-1, float32
    """
    if not os.path.isfile(fn):
        raise ValueError("No file {0}".format(fn))
    img = cv2.imread(fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize to given size
    if shape is not None:
        img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_CUBIC)

    # resize to have div32 size
    if crop_div32:
        new_size = (img.shape[0] - img.shape[0] % 32,
                    img.shape[1] - img.shape[1] % 32)

        img = img[(img.shape[0] - new_size[0]) // 2:(img.shape[0] + new_size[0]) // 2,
              (img.shape[1] - new_size[1]) // 2:(img.shape[1] + new_size[1]) // 2]

    return (img / 255.).astype(np.float32)


def get_noise_image(img_orig, sigma=25):
    """
    Add Gaussian noise to input image.
    :param img_orig: original image
    :param sigma: std of Gaussian distribution (noise) - Note: if 0-1 range, sigma/255.
    :return: noisy image with RGB, range 0-1, float32
    """
    return np.clip(img_orig + np.random.normal(scale=sigma, size=img_orig.shape), 0, 1).astype(np.float32)


def get_low_resolution_image(img_orig, factor=4, interpolation=cv2.INTER_AREA):
    """
    Get low resolution image of high resolution image
    :param img_orig: original image
    :param factor: factor used for downscaling (default: 4)
    :param interpolation: interpolation method (default: INTER_AREA)
    :return:
    """
    lr_size = (img_orig.shape[1] // factor, img_orig.shape[0] // factor)
    img_lr = cv2.resize(img_orig, dsize=lr_size, interpolation=interpolation)
    return img_lr


def get_baseline_resized_images(img_lr, shape):
    """
    Upscale image with cubic and nearest neighbor method
    Generates the baseline images for super resolution task
    :param img_lr: original image
    :param shape: shape of high resolution image
    :return: upscaled images with bicubic and nearest neighbor interpolation method, range 0-1
    """
    shape = (shape[1], shape[0])
    img_bicubic = cv2.resize(img_lr, dsize=shape, interpolation=cv2.INTER_CUBIC).clip(0, 1)
    img_nearest = cv2.resize(img_lr, dsize=shape, interpolation=cv2.INTER_NEAREST).clip(0, 1)
    return img_bicubic, img_nearest


def get_network_input(shape=None, var=1. / 10, seed=13317):
    """
    Generate a tensor with given shape filled with uniform noise
    :param shape: tensor shape (B, H, W, C) (default: [1,512,512,32])
    :param var: variable for multiplication (default: 1/10)
    :param seed: random seed init
    :return: uniform noise tensor
    """
    if shape is None:
        shape = [1, 512, 512, 32]
    return tf.random.uniform(shape, seed=seed) * var


def convert_to_tf_tensor(img):
    """
    Converts a np array to tf tensor with batch size 1
    :param img: image to be converted to tf tensor
    :return: tf tensor with shape (B, H, W, C)
    """
    if isinstance(img, tf.Tensor):
        raise ValueError("Img is already a tf tensor.")
    return tf.convert_to_tensor(img.reshape(1, *img.shape))


def convert_to_numpy_array(tensor):
    """
    Converts a tf tensor to a np array without batch size
    :param tensor: tensor to be converted to np array
    :return: np array with shap (H, W, C)
    """
    if isinstance(tensor, np.ndarray):
        raise ValueError("Tensor is already a np.ndarray.")
    return tf.make_ndarray(tf.make_tensor_proto(tensor))[0]


def create_cutout(images, cut_out_region):
    """
    Create cut outs of images
    :param images:  list of images
    :param cut_out_region: boundaries for cut out region in list with
    [x_top_left, y_top_left, x_bottom_right, y_bottom_left]
    :return: list of images with cut outs
    """
    result = []
    for img in images:
        result.append(img[cut_out_region[1]:cut_out_region[3], cut_out_region[0]:cut_out_region[2]])
    return result



