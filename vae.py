import os
import time
import glob
import PIL
import imageio

from IPython import display

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32")

train_images /= 255.0
test_images /= 255.0

train_images[train_images >= 0.5] = 1.0
train_images[train_images < 0.5] = 0.0

test_images[test_images >= 0.5] = 1.0
test_images[test_images < 0.5] = 0.0


TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential([
            layers.InputLayer(input_shape=[28, 28, 1]),
            layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation="relu"),

            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim)
        ])

        self.generative_net = tf.keras.Sequential([
            layers.InputLayer(input_shape=[latent_dim]),

            layers.Dense(7*7*32, activation="relu"),
            layers.Reshape([7, 7, 32]),

            layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same", activation="relu"),

            layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding="same", activation="relu"),
        ])

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))

        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return mean + tf.exp(logvar * 0.5) * eps

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)

    return tf.reduce_sum(-0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) +
        logvar + log2pi), axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)

    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

epochs = 100
latent_dim = 50
num_examples_to_generate = 16

random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, ..., 0], cmap="gray")
        plt.axis("off")

    plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    # plt.show()

generate_and_save_images(model, 0, random_vector_for_generation)

for epoch in range(1, epochs + 1):
    start_time = time.time()

    for train_x in train_dataset:
        gradients, loss = compute_gradients(model, train_x)
        apply_gradients(optimizer, gradients, model.trainable_variables)

    end_time = time.time()

    if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()

        for test_x in test_dataset:
            loss(compute_loss(model, test_x))

        elbo = -loss.result()
        display.clear_output(wait=False)

        print("Epoch: {}, Test set ELBO: {}, "
              "time elapsed for current epoch {}".format(epoch, elbo, end_time - start_time))

        generate_and_save_images(model, epoch, random_vector_for_generation)
