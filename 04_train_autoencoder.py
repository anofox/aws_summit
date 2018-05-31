from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
from showcase.models import AutoencoderModel
from showcase.image_sequence import create_rgbdseq_from_files


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    return
    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# data set
image_height = 96
image_width = 128
n_chans = 4
batch_size = 128
input_shape = (image_height, image_width, n_chans)

train_gen = create_rgbdseq_from_files(glob_pattern="data/*.npz", batch_size=batch_size, is_validation=False)
val_gen = create_rgbdseq_from_files(glob_pattern="data/*.npz", batch_size=batch_size, is_validation=True)

x_train = np.vstack(filter(lambda x: isinstance(x, bool)==False, [train_gen[i] for i in range(600)]))
x_test = np.vstack(filter(lambda x: isinstance(x, bool)==False, [val_gen[i] for i in range(80)]))


model = AutoencoderModel(input_shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", help="Load h5 model trained weights")
    parser.add_argument("-e", "--epochs", default=25)

    args = parser.parse_args()

    if args.weights:
        model.load_model(args.weights)
    else:
        # train the autoencoder
        hist = model.fit(x_train, x_test)
        model.save_model('autoenc_cnn_rgbd_180526.h5')
        plot_history(hist)

    #plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")