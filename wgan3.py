import os
import argparse

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

import keras
import keras.backend as K

from keras import initializers
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, LeakyReLU


def save_img(samples, n, path):
    """Saves the generated samples as a PNG.
    `samples` is the numpy array of generated outputs.
    `n` is the epoch (to include in the filename).
    `path` is the path where the file is saved."""

    img_rows = []
    for c in range(10):
        img_rows.append(np.concatenate(samples[c * 10:(1 + c) * 10]).reshape(280, 28))
    img = np.hstack(img_rows)

    fname = path + '/samples_%07d.png' % n
    plt.imsave(fname, img, cmap=plt.cm.gray)

    return img


def get_data():
    """Returns the preprocessed MNIST dataset as a matrix of vectors where each row is
    a normalized image (pixels between 0 and 1)."""

    # loads the dataset (we don't care about a test set so we join it)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate((X_train, X_test), axis=0).astype('float32')

    # normalize the pixels
    X /= 255

    # flatten
    X = X.reshape((70000, 784))

    assert X.shape == (70000, 784), X.shape

    return X


def mean(y_true, y_pred):
    """MEan loss function (used for the Wasserstain loss)."""

    return K.mean(y_true * y_pred)


def get_generator(input_size, output_size):
    """Returns the generator model `G`.
    `input_size` and `output_size` are the size of the input vectors z for `G` and the output of `G`.
    """
    xavier = initializers.glorot_normal()

    z = Input(shape=(input_size,), name='input_z')
    a = Dense(128, kernel_initializer=xavier)(z)
    a = LeakyReLU()(a)
    a = Dense(256, kernel_initializer=xavier)(z)
    a = LeakyReLU()(a)
    a = Dense(output_size, activation='sigmoid', kernel_initializer=xavier)(a)

    G = Model(inputs=[z], outputs=[a], name='G')
    return G


def get_discriminator(input_size, G, G_input_size):
    """Returns the discriminator networks `D` and `DG`.
    `input_size` is the input size of `D` and `DG`.
    `G` is the generator model.
    
    Return a tuple of 2 elements:
      * The first element is `D`, the discriminator/critic
      * The second element id `DG` -> D(G(z))"""
    
    xavier = initializers.glorot_normal()

    x = Input(shape=(input_size,), name='input_x')
    a = Dense(256, kernel_initializer=xavier)(x)
    a = LeakyReLU()(a)
    a = Dropout(0.5)(a)
    a = Dense(256, kernel_initializer=xavier)(a)
    a = LeakyReLU()(a)
    a = Dropout(0.5)(a)
    a = Dense(16, kernel_initializer=xavier)(a)
    a = LeakyReLU()(a)
    a = Dropout(0.5)(a)

    # creates an output to determine if an input is fake
    is_fake = Dense(1, activation='linear', name='output_is_fake')(a)

    # creates a D model that receives a real example as input
    D = Model(inputs=[x], outputs=[is_fake], name='D')

    # creates another model that uses G as input
    z = Input(shape=(G_input_size,), name='D_input_z')
    is_fake = D(G(inputs=[z]))
    DG = Model(inputs=[z], outputs=[is_fake], name='DG')

    # D shouldn't be trained during the generator's training faze
    DG.get_layer('D').trainable = False
    DG.compile(optimizer=RMSprop(lr=1e-3), loss=[mean])

    D.trainable = True
    D.compile(optimizer=RMSprop(lr=1e-3), loss=[wasserstein])

    return D, DG


def train(D, G, DG, X, z_size, epochs, batch_size=32, samples_path='.'):
    """Trains the GAN for the given number of `epochs`.
    `D` is the discriminator, `G` the generator and `DG` the discriminator receiving as
    input the output of `G` -> D(G(z))
    `z_size` is the size of the input vector for `G`
    `samples_path` is the path where the output images are stored.
    """
    
    def get_z_batch(batch_size=batch_size):
        """Generates a batch of z vectors to use as input for `G`"""

        z = np.random.normal(0., 1., (batch_size, z_size))
        return z

    # gets a fixed Z vector to output samples during the training
    z_test = get_z_batch(batch_size=100)

    progress_bar = Progbar(target=epochs)

    DG_losses = []
    D_real_losses = []
    D_fake_losses = []

    for i in range(epochs):
        # train D
        for _ in range(5):
            # clip D weights
            for l in D.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -0.1, 0.1) for w in weights]
                l.set_weights(weights)

            # sample a batch of real data
            index = np.random.choice(len(X), batch_size, replace=False)
            real_batch_x = X[index]

            # train on real data
            # the labels are negated (minus sign) so convert the problem into a minimization
            loss = D.train_on_batch(real_batch_x, [-np.ones(batch_size)])
            D_real_losses.append(loss)

            # generate fake data and train on it too
            z = get_z_batch()
            fake_batch = G.predict(z)

            loss = D.train_on_batch(fake_batch, [np.ones(batch_size)])
            D_fake_losses.append(loss)

        # train G
        z = get_z_batch()

        loss = DG.train_on_batch(z, [-np.ones(batch_size)])
        DG_losses.append(loss)

        if len(D_real_losses) > 0:
            progress_bar.update(
                i,
                values=[
                    ('D_real', np.mean(D_real_losses[-5:], axis=0)),
                    ('D_fake', np.mean(D_fake_losses[-5:], axis=0)),
                    ('D(G)', np.mean(DG_losses[-5:], axis=0)),
                ]
            )

        # shows a generated image
        if i % 1 == 0 and samples_path is not None:
            samples = G.predict(z_test)
            fig = save_img(samples, i, path=samples_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true',
                        help='Run the training using GPU')
    parser.add_argument('epochs', type=int,
                        help='Number of epochs to run the training')
    parser.add_argument('--output', help='Directory where the generated samples are saved', default='.')
    args = parser.parse_args()

    if args.gpu:
        print 'Using GPU'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    np.random.seed(0)

    K.set_image_dim_ordering('tf')

    G_INPUT_SIZE = 20 + 10

    # gets the preprocessed MNIST dataset
    X = get_data()

    # gets the models
    G = get_generator(input_size=G_INPUT_SIZE, output_size=784)
    D, DG = get_discriminator(input_size=784, G=G, G_input_size=G_INPUT_SIZE)

    train(D, G, DG, X, G_INPUT_SIZE, epochs=args.epochs, samples_path=args.output)
