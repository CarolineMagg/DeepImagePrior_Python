####################################################################
# DIP training loop implementation
####################################################################

import os
import copy
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio

from src.utils.data_plotting import plot_images
from src.utils.data_processing import get_measurements

__author__ = "c.magg"


class DIPTraining:
    """
    Custom Training loop for DIP training
    """

    def __init__(self, model, loss, opt, seed=13317):
        """
        Constructor
        :param model: model to train
        :param loss: loss method
        :param opt: optimizer
        """
        self._model = model
        self._loss = loss
        self._opt = opt

        self.num_iter = None
        self.psrn_noisy_last = 0
        self.reg_noise_std = 0
        self.exp_weight = None

        self.plot_info = True
        self.print_info = True
        self.save_history = True
        self.save_model = True

        self._loss_values = []
        self._psnr_corrupted = []
        self._psnr_gt = []
        self._psnr_smooth = []
        self._checkpoint = "train_run/cp.ckpt"
        self._monitor = "psnr_gt"
        self._each_epoch = 100
        self._epoch = None

        tf.random.set_seed(seed)

    def reset_history(self):
        """
        Reset training history information
        :return:
        """
        self._loss_values = []
        self._psnr_corrupted = []
        self._psnr_gt = []
        self._psnr_smooth = []

    def set_callbacks(self, plot_info=True, print_info=True, save_history=True, save_model=True,
                      checkpoint="train_run/cp.ckpt", monitor="psnr_gt", plot_every_epoch=100):
        """
        Set callback flags for training
        :param plot_info: plot output image in a regular manner
        :param print_info: print each epoch the loss value and PSNR
        :param save_history: save the loss value and psnr history
        :param save_model: save model
        :param checkpoint: name for model checkpoint file
        :param monitor: monitor measurements for model save checkpoint
        :param plot_every_epoch: number of epoch where plot information is invoked
        :return:
        """
        self.plot_info = plot_info
        self.print_info = print_info
        self.save_history = save_history
        self.save_model = save_model
        self._checkpoint = checkpoint
        self._monitor = monitor
        self._each_epoch = plot_every_epoch

    def train_denoising(self, net_input, img, img_gt=None, num_iter=3000, reg_noise_std=1. / 30., exp_weight=0.99):
        """
        Start denoising training
        :param net_input: net input with same shape as first model layer
        :param img: corrupted image used for loss calculation
        :param img_gt: ground truth image used for PSNR calculation
        :param num_iter: number of iterations
        :param reg_noise_std: std of regularisation term for noise
        :param exp_weight: smoothing factor
        :return:
        """

        self.num_iter = num_iter
        self.reg_noise_std = reg_noise_std
        self.exp_weight = exp_weight
        net_input_saved = copy.deepcopy(net_input)
        shape = net_input.shape
        output_avg = None

        self.reset_history()

        if not isinstance(net_input, tf.Tensor):
            raise ValueError("Training input need to be tf tensors.")

        for epoch in range(num_iter):
            self._epoch = epoch
            if self.reg_noise_std > 0:  # add again noise to input image
                net_input = net_input_saved + (tf.random.normal(shape) * self.reg_noise_std)

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
                output = self._model(net_input, training=True)

                # Compute the loss value for corrupted image and model output
                loss_value = self._loss(img, output)

            # smoothing
            if output_avg is None:
                output_avg = copy.deepcopy(output)
            else:
                output_avg = output_avg * exp_weight + output * (1 - exp_weight)

            output_np = tf.make_ndarray(tf.make_tensor_proto(output))[0]
            output_avg_np = tf.make_ndarray(tf.make_tensor_proto(output_avg))[0] if output_avg is not None else None

            # Calculate PSNR
            psnr = self.psnr_comparison(output_np, img, output_avg=output_avg_np, img_gt=img_gt)

            # Perform callbacks
            self.perform_callbacks(loss_value, psnr, output_np, output_avg_np)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, self._model.trainable_weights)

            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            self._opt.apply_gradients(zip(grads, self._model.trainable_weights))

    def train_super_resolution(self, net_input, img, img_gt=None, num_iter=3000, reg_noise_std=1. / 30., exp_weight=0.99):
        """
        Start super resolution training
        :param net_input: net input with same shape as first model layer
        :param img: corrupted image used for loss calculation
        :param img_gt: ground truth image used for PSNR calculation
        :param num_iter: number of iterations
        :param reg_noise_std: std of regularisation term for noise
        :param exp_weight: smoothing factor
        :return:
        """

        self.num_iter = num_iter
        self.reg_noise_std = reg_noise_std
        self.exp_weight = exp_weight
        net_input_saved = copy.deepcopy(net_input)
        shape = net_input.shape
        shape_lr = img.shape[0], img.shape[1]
        output_avg = None

        self.reset_history()

        if not isinstance(net_input, tf.Tensor):
            raise ValueError("Training input need to be tf tensors.")

        for epoch in range(num_iter):
            self._epoch = epoch
            if self.reg_noise_std > 0:  # add again noise to input image
                net_input = net_input_saved + (tf.random.normal(shape) * self.reg_noise_std)

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
                output = self._model(net_input, training=True)

                # Downsample input image to fit corrupted image
                output_lr = tf.image.resize(output, shape_lr, method="lanczos3")

                # Compute the loss value for corrupted image and downsampled model output
                loss_value = self._loss(img, output_lr)

            # Smoothing
            if output_avg is None:
                output_avg = copy.deepcopy(output)
            else:
                output_avg = output_avg * exp_weight + output * (1 - exp_weight)

            output_np = tf.make_ndarray(tf.make_tensor_proto(output))[0]
            output_lr_np = tf.make_ndarray(tf.make_tensor_proto(output_lr))[0]
            output_avg_np = tf.make_ndarray(tf.make_tensor_proto(output_avg))[0] if output_avg is not None else None

            # Calculate PSNR
            psnr = self.psnr_comparison(output_lr_np, img, output_for_gt=output_np, output_avg=output_avg_np, img_gt=img_gt)

            # Perform callbacks
            self.perform_callbacks(loss_value, psnr, output_np, output_avg_np)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, self._model.trainable_weights)

            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            self._opt.apply_gradients(zip(grads, self._model.trainable_weights))

    def perform_callbacks(self, loss_value, psnr, output_np, output_avg_np):
        """
        Perform callbacks in training run
        :return:
        """
        if self.print_info:
            self.print_information(self._epoch, loss_value, psnr)
        if self.plot_info:
            self.plot_images(self._epoch, output_np, output_avg_np, self._each_epoch)
        if self.save_history:
            self.save_history_values(loss_value, psnr)
            if self.save_model:
                self.save_model_checkpoint()

    @staticmethod
    def psnr_comparison(output, img, output_for_gt=None, output_avg=None, img_gt=None):
        """
        Compare network output with corrupted image and compute PNSR
        :param output: model output with same shape as img
        :param img: corrupted image
        :param output_for_gt: model output with same shape as img_gt, if None then same as output
        :param output_avg: smooth model output with same shape as img_gt
        :param img_gt: ground truth image
        :return: dict with PSNR for corrupted, ground truth and smooth ground truth images
        """
        if output_for_gt is None:
            output_for_gt = output
        return {"psnr_corrupted": peak_signal_noise_ratio(img, output),
                "psnr_gt": peak_signal_noise_ratio(img_gt, output_for_gt) if img_gt is not None else None,
                "psnr_gt_smooth": peak_signal_noise_ratio(img_gt,
                                                          output_avg) if img_gt is not None and output_avg is not None else None}

    @staticmethod
    def print_information(epoch, loss_value, psrn):
        print('Iteration %05d    Loss %f   PSNR_corrupted: %f   PSRN_gt: %f PSNR_gt_sm: %f' %
              (epoch, loss_value.numpy(), psrn["psnr_corrupted"], psrn["psnr_gt"], psrn["psnr_gt_smooth"]), '\r',
              end='')

    @staticmethod
    def plot_images(epoch, output, output_avg, each_epoch=100):
        if epoch % each_epoch == 0:
            plot_images([output, output_avg])

    def save_history_values(self, loss_value, psnr):
        """
        Save history information
        :param loss_value: loss value
        :param psnr: dict with psnr values
        :return:
        """
        self._loss_values.append(loss_value)
        self._psnr_corrupted.append(psnr["psnr_corrupted"])
        self._psnr_gt.append(psnr["psnr_gt"])
        self._psnr_smooth.append(psnr["psnr_gt_smooth"])

    def save_model_checkpoint(self, debug=False):
        """
        Save model if better than before.
        Works only if history values are saved.
        :param debug: flag for plotting information about saving model checkpoint (default: False - off)
        :return:
        """
        if self._monitor == "psnr_gt":
            if self._psnr_gt[self._epoch] > self._psnr_gt[self._epoch - 1]:
                if debug:
                    print("\nIteration %05d    PSNR_GT improved from %f to %f. Save model to %s" %
                          (self._epoch, self._psnr_gt[self._epoch], self._psnr_gt[self._epoch - 1], self._checkpoint.format(epoch=self._epoch)),
                          end="\n")
                self._model.save_weights(self._checkpoint)

        elif self._monitor == "psnr_corrupted":
            if self._psnr_corrupted[self._epoch] > self._psnr_corrupted[self._epoch - 1]:
                if debug:
                    print("\nIteration %05d    PSNR_corrupted improved from %f to %f. Save model to %s" %
                          (self._epoch, self._psnr_corrupted[self._epoch], self._psnr_corrupted[self._epoch - 1],
                           self._checkpoint.format(epoch=self._epoch)),
                          end="\n")
                self._model.save_weights(self._checkpoint)

        else:
            if self._loss_values[self._epoch] > self._loss_values[self._epoch - 1]:
                if debug:
                    print("\nIteration %05d    Loss improved from %f to %f. Save model to %s" %
                          (self._epoch, self._loss_values[self._epoch], self._loss_values[self._epoch - 1],
                           self._checkpoint.format(epoch=self._epoch)),
                          end="\n")
                self._model.save_weights(self._checkpoint)

    def plot_history(self, store=False):
        """
        Create plots of history for loss and PSNR.
        Works only if save history was active.
        :param store: flag if the plot should be saved.
        :return:
        """
        if self._loss_values is not None and len(self._loss_values) > 0:
            fig, axs = plt.subplots(2)
            axs[0].plot(range(len(self._loss_values)), [val.numpy() for val in self._loss_values])
            axs[0].set_title("Loss values")

            axs[1].plot(range(len(self._psnr_gt)), self._psnr_gt, "-b", label="PSNR_gt")
            axs[1].plot(range(len(self._psnr_corrupted)), self._psnr_corrupted, '-r', label="PSNR_corrupted")
            axs[1].plot(range(len(self._psnr_smooth)), self._psnr_smooth, '-g', label="PSNR_gt_smooth")
            axs[1].legend()
            axs[1].set_title("PSNR")

            if store:
                fn = os.path.join("/".join(self._checkpoint.split('/')[:-1]), "plot.png")
                fig.savefig(fn)
                print("Store plot in {0}".format(fn))
            else:
                plt.show()
        else:
            raise ValueError("No history information available.")

    def evaluate(self, net_input, img_corrupted, img_gt=None, store=False):
        """
        Evaluate current model
        :param net_input: network input
        :param img_corrupted: corrupted image
        :param img_gt: ground truth image
        :param store: flag if the result should be stored or not
        :return:
        """
        output = self._model(net_input)
        output_np = tf.make_ndarray(tf.make_tensor_proto(output))[0]
        output_lr = None

        if output_np.shape != img_corrupted.shape:
            shape_lr = img_corrupted.shape[0], img_corrupted.shape[1]
            output_lr = tf.image.resize(output, shape_lr, method="lanczos3")

        metrics = get_measurements(img_gt, output_np)
        if output_lr is not None:
            metrics_corrupted = get_measurements(img_corrupted, output_lr)
        else:
            metrics_corrupted = get_measurements(img_corrupted, output_np)

        print("Results:")
        print(" Ground truth:")
        for k, v in metrics.items():
            print("  {0}: {1} ".format(k, v))
        print(" Corrupted data:")
        for k, v in metrics_corrupted.items():
            print("  {0}: {1} ".format(k, v))

        if store:
            fn = os.path.join("/".join(self._checkpoint.split('/')[:-1]), "results.csv")
            df = pd.DataFrame(columns=["MSE", "PSNR", "SSIM"], index=["GT", "Corrupted"])
            df.loc["GT"] = list(metrics.values())
            df.loc["Corrupted"] = list(metrics_corrupted.values())
            df.to_csv(fn)
            print("Store results in {0}".format(fn))




