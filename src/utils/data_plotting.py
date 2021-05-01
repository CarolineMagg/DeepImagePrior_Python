####################################################################
# Plotting and debugging methods for data handling:
# * plot list of images
# * plot cut out images
# * print image debug information (dtype, type, shape, pixel range)
####################################################################

import matplotlib.pyplot as plt
from src.utils.data_processing import *

__author__ = "c.magg"


def plot_images(images, figsize=(15, 15), axis=False, title=None):
    """
    Plotting method for list of im
    :param images: list of images
    :param figsize: figure size for plot
    :param axis: flag if axis should be shown or not
    :return:
    """
    nr = len(images)
    fig = plt.figure(figsize=figsize)
    for idx in range(nr):
        plt.subplot(eval("1" + str(nr) + str(idx + 1)))
        plt.axis('off') if axis == False else plt.axis('on')
        plt.imshow(images[idx])
    if title is not None:
        print(title)
    plt.show()


def plot_cut_outs(images, cut_out_region):
    """
    Create cut outs and plot images
    :param images: list of images
    :param cut_out_region: boundaries for cut out region in list with
    [x_top_left, y_top_left, x_bottom_right, y_bottom_left]
    :return:
    """
    cut_outs = create_cutout(images, cut_out_region)
    toshow = []
    for img in images:
        toshow_single = img.copy()
        toshow_single = cv2.rectangle(toshow_single, (cut_out_region[0], cut_out_region[1]),
                                      (cut_out_region[2], cut_out_region[3]), (0, 0, 0), 2)
        toshow.append(toshow_single)

    plot_images(images, title="Images")
    plot_images(toshow, title="Images with cut outs")
    plot_images(cut_outs, title="Cut outs")


def plot_cut_outs_different_sizes(images, images_lr, cut_out_region):
    """
    Create cut outs and plot images for images with different sizes
    :param images: list of images
    :param cut_out_region: boundaries for cut out region in list with
    [x_top_left, y_top_left, x_bottom_right, y_bottom_left] for images
    :param images_lr: list of images with other shape
    :return:
    """
    shape = get_height_width(images[0].shape)
    shape2 = get_height_width(images_lr[0].shape)
    # recalculate cut out region for other dimension
    cut_out_region2 = [int(cut_out_region[0]/shape[1]*shape2[1]), int(cut_out_region[1]/shape[0]*shape2[0]),
                       int(cut_out_region[2]/shape[1]*shape2[1]), int(cut_out_region[3]/shape[0]*shape2[0])]
    cut_outs = create_cutout(images, cut_out_region)
    cut_outs2 = create_cutout(images_lr, cut_out_region2)
    toshow = []
    for img in images:
        toshow_single = img.copy()
        toshow_single = cv2.rectangle(toshow_single, (cut_out_region[0], cut_out_region[1]),
                                      (cut_out_region[2], cut_out_region[3]), (0, 0, 0), 2)
        toshow.append(toshow_single)
    for img in images_lr:
        toshow_single = img.copy()
        toshow_single = cv2.rectangle(toshow_single, (cut_out_region[0], cut_out_region[1]),
                                      (cut_out_region[2], cut_out_region[3]), (0, 0, 0), 2)
        toshow.append(toshow_single)

    plot_images(images+images_lr, title="Images")
    plot_images(toshow, title="Images with cut outs")
    plot_images(cut_outs+cut_outs2, title="Cut outs")


def get_height_width(orig_shape):
    """
    Get height and weight of shape tuple
    :param orig_shape: original shape with different formats:
    * (B,H,W,C)
    * (H,W,C)
    * (H,W)
    :return: height and width of shape
    """
    shape = None
    if len(orig_shape) == 2:  # (H, W)
        shape = orig_shape
    elif len(orig_shape) == 3:  # (H, W, C)
        shape = orig_shape[0], orig_shape[1]
    elif len(orig_shape) == 4:  # (B, H, W, C)
        shape = orig_shape[1], orig_shape[2]
    return shape


def image_information(images):
    """
    Provide (debug) information of image
    :param images: (list of) input image
    """
    if not isinstance(images, list):
        images = [images]
    for img in images:
        print("dtype: ", img.dtype, ", type: ", type(img))
        print("size: ", np.shape(img), ", range: {0} - {1}".format(np.min(img), np.max(img)))
