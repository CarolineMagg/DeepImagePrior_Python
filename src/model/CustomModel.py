####################################################################
# CustomModel implementation to customize fit method
####################################################################
import tensorflow as tf


class CustomModel(tf.keras.Model):

    reg_noise_std = 1. / 30.
    downscaling_method = "lanczos3"

    def train_step(self, data):
        x, y = data
        if self.reg_noise_std > 0:  # add again noise to input image
            x = x + (tf.random.normal([1, *x.shape[1:]]) * self.reg_noise_std)

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            # Resize if necessary
            if y_pred[0].shape != y[0].shape:
                y_pred = tf.image.resize(y_pred, [*y.shape[1:3]], method=self.downscaling_method)
            # Compute loss
            loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)(y[0], y_pred[0])

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute metrics
        psnr = tf.image.psnr(y_pred[0], y[0], max_val=1.0)
        ssim = tf.image.ssim(y_pred[0], y[0], max_val=1.0)
        return {"loss": loss, "psnr": psnr, "ssim": ssim}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        if y_pred[0].shape != y[0].shape:
            y_pred = tf.image.resize(y_pred, [*y.shape[1:3]], method=self.downscaling_method)
        loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)(y, y_pred)
        psnr = tf.image.psnr(y_pred[0], y[0], max_val=1.0)
        ssim = tf.image.ssim(y_pred[0], y[0], max_val=1.0)
        return {"loss": loss, "psnr": psnr, "ssim": ssim}

    @property
    def metrics_names(self):
        return ["loss", "psnr", "ssim"]

