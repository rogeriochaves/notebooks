import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


inputs_ = tf.placeholder(tf.float32, [None, 28, 28, 1])
targets_ = tf.placeholder(tf.float32, [None, 28, 28, 1])


def lrelu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


# Encoder
with tf.name_scope('en-convolutions'):
    conv1 = tf.layers.conv2d(inputs_, filters=32, kernel_size=(3, 3), strides=(
        1, 1), padding='SAME', use_bias=True, activation=lrelu, name='conv1')
# Now 28x28x32
with tf.name_scope('en-pooling'):
    maxpool1 = tf.layers.max_pooling2d(
        conv1, pool_size=(2, 2), strides=(2, 2), name='pool1')
# Now 14x14x32
with tf.name_scope('en-convolutions'):
    conv2 = tf.layers.conv2d(maxpool1, filters=32, kernel_size=(3, 3), strides=(
        1, 1), padding='SAME', use_bias=True, activation=lrelu, name='conv2')
# Now 14x14x32
with tf.name_scope('encoding'):
    encoded = tf.layers.max_pooling2d(
        conv2, pool_size=(2, 2), strides=(2, 2), name='encoding')
# Now 7x7x32.
# latent space

# Decoder
with tf.name_scope('decoder'):
    conv3 = tf.layers.conv2d(encoded, filters=32, kernel_size=(3, 3), strides=(
        1, 1), name='conv3', padding='SAME', use_bias=True, activation=lrelu)
# Now 7x7x32
    upsample1 = tf.layers.conv2d_transpose(
        conv3, filters=32, kernel_size=3, padding='same', strides=2, name='upsample1')
# Now 14x14x32
    upsample2 = tf.layers.conv2d_transpose(
        upsample1, filters=32, kernel_size=3, padding='same', strides=2, name='upsample2')
# Now 28x28x32
    logits = tf.layers.conv2d(upsample2, filters=1, kernel_size=(
        3, 3), strides=(1, 1), name='logits', padding='SAME', use_bias=True)
# Now 28x28x1
# Pass logits through sigmoid to get reconstructed image
    decoded = tf.sigmoid(logits, name='recon')

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets_)

learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(loss)  # cost
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # optimizer


# Training

sess = tf.Session()
# tf.reset_default_graph()

saver = tf.train.Saver()
loss = []
valid_loss = []


display_step = 1
epochs = 25
batch_size = 64
#lr=[1e-3/(2**(i//5))for i in range(epochs)]
lr = 1e-5
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./graphs', sess.graph)
for e in range(epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    for ibatch in range(total_batch):
        batch_x = mnist.train.next_batch(batch_size)
        batch_test_x = mnist.test.next_batch(batch_size)
        imgs_test = batch_x[0].reshape((-1, 28, 28, 1))
        noise_factor = 0.5
        x_test_noisy = imgs_test + noise_factor * \
            np.random.normal(loc=0.0, scale=1.0, size=imgs_test.shape)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)
        imgs = batch_x[0].reshape((-1, 28, 28, 1))
        x_train_noisy = imgs + noise_factor * \
            np.random.normal(loc=0.0, scale=1.0, size=imgs.shape)
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: x_train_noisy,
                                                         targets_: imgs, learning_rate: lr})

        batch_cost_test = sess.run(cost, feed_dict={inputs_: x_test_noisy,
                                                    targets_: imgs_test})
    if (e + 1) % display_step == 0:
        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Training loss: {:.4f}".format(batch_cost),
              "Validation loss: {:.4f}".format(batch_cost_test))

    loss.append(batch_cost)
    valid_loss.append(batch_cost_test)
    plt.plot(range(e + 1), loss, 'bo', label='Training loss')
    plt.plot(range(e + 1), valid_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()
    saver.save(sess, 'encode_model')

batch_x = mnist.test.next_batch(10)
imgs = batch_x[0].reshape((-1, 28, 28, 1))
noise_factor = 0.5
x_test_noisy = imgs + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=imgs.shape)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
recon_img = sess.run([decoded], feed_dict={inputs_: x_test_noisy})[0]
plt.figure(figsize=(20, 4))
plt.title('Reconstructed Images')
print("Original Images")
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(imgs[i, ..., 0], cmap='gray')
plt.show()
plt.figure(figsize=(20, 4))
print("Noisy Images")
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test_noisy[i, ..., 0], cmap='gray')
plt.show()
plt.figure(figsize=(20, 4))
print("Reconstruction of Noisy Images")
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(recon_img[i, ..., 0], cmap='gray')
plt.show()

writer.close()

sess.close()
