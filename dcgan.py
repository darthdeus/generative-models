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

def make_generator_model():
    model = tf.keras.Sequential()

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

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# plt.imshow(generated_image[0, ..., 0], cmap="gray")
# plt.show()

def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    loss = real_loss + fake_loss

    return loss

def generator_loss(fake_output):
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
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, Z_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    generator_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_grad, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_grad, discriminator.trainable_variables))


def train(dataset, epochs, start_epoch = 0):
    for epoch in range(start_epoch, start_epoch + epochs):
        start = time.time()

        for image_batch in dataset:
            print(".", end="", flush=True)
            train_step(image_batch)

        display.clear_output(wait=True)

        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            print("Checkpoint saved", file=sys.stderr)
            checkpoint.save(file_prefix = checkpoint_prefix)

        print("Time for epoch {} is {}".format(epoch + 1, time.time() - start))

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
