import sys
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow.keras.layers as layers
import time

from IPython import display
import tensorflow as tf

(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(len(train_images), 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 256
Z_DIM = 100

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)



def make_generator_model(dcgan: bool):
    model = tf.keras.Sequential()

    if dcgan:
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(Z_DIM,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False))
        assert model.output_shape == (None, 28, 28, 1)
    else:
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(Z_DIM,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(28*28, use_bias=False))
        model.add(layers.Reshape((28, 28, 1)))

    return model

def make_discriminator_model(dcgan: bool):
    model = tf.keras.Sequential()

    if dcgan:
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
    else:
        model.add(layers.Dense(512, input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU())

    model.add(layers.Dense(1))

    return model


USE_DCGAN = True
USE_WGAN_LOSS = True

generator = make_generator_model(USE_DCGAN)
discriminator = make_discriminator_model(USE_DCGAN)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output, wgan_loss: bool):
    if wgan_loss:
        return tf.reduce_mean(real_output - fake_output)
    else:
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

        loss = real_loss + fake_loss

        return loss

def generator_loss(fake_output, wgan_loss: bool):
    if wgan_loss:
        return tf.reduce_mean(fake_output)
    else:
        return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = "./checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 500
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, Z_DIM])

@tf.function
def train_step(critic_batches, generator_batches):
    for critic_images in critic_batches:
        noise = tf.random.normal([BATCH_SIZE, Z_DIM])

        with tf.GradientTape() as critic_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(critic_images, training=True)
            fake_output = discriminator(generated_images, training=True)

            disc_loss = discriminator_loss(real_output, fake_output, USE_WGAN_LOSS)

        discriminator_grad = critic_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_grad, discriminator.trainable_variables))

        for variable in discriminator.trainable_variables:
            tf.clip_by_value(variable, -0.01, 0.01)

    # TODO: unnecessary, we don't need reals for training G
    for generator_images in generator_batches:
        noise = tf.random.normal([BATCH_SIZE, Z_DIM])

        with tf.GradientTape() as gen_tape:
            generated_images = generator(noise, training=True)

            # real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output, USE_WGAN_LOSS)

        generator_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_grad, generator.trainable_variables))


def train(dataset, epochs, start_epoch = 0):
    train_start = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start = time.time()

        critic_iters = 5

        iterator = iter(dataset)

        while True:
            try:
                critic_batches = [next(iterator) for _ in range(critic_iters)]
                generator_batches = [next(iterator)]

                print(".", end="", flush=True)

                # TODO: unshuffle & take N + 1 separately?
                train_step(critic_batches, generator_batches)
            except StopIteration:
                break

        display.clear_output(wait=True)

        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            print("Checkpoint saved", file=sys.stderr)
            checkpoint.save(file_prefix = checkpoint_prefix)

        curr_time = time.time()
        print("{:04d} - T (total): {:.2f}\tT/epoch: {:.2f}".format(epoch + 1, curr_time - train_start, curr_time - epoch_start))

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(len(predictions)):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, ..., 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    plt.savefig("samples/image_at_epoch{:04d}.png".format(epoch))
    # plt.show()

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

filenames = sorted(glob.glob("samples/*.png"))

if len(filenames) > 0:
    with imageio.get_writer("dcgan.gif", mode="I") as writer:
        last = -1

        for i, filename in enumerate(filenames):
            frame = 2*(i**0.5)

            if round(frame) > round(last):
                last = frame
            else:
                continue

            image = imageio.imread(filename)
            writer.append_data(image)

        image = imageio.imread(filename)
        writer.append_data(image)

    last_epoch = int(filenames[-1].replace("samples/image_at_epoch", "").replace(".png", "")) + 1
    print("GIF generated")
else:
    last_epoch = 0

train(train_dataset, EPOCHS, start_epoch = last_epoch)
