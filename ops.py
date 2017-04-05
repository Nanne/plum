
"""Couple of layers implemented in tensorflow."""
import tensorflow as tf

def dense(batch_input, out_dim):
    """Add dense layer to the graph."""
    with tf.name_scope('dense'):
        in_dim = batch_input.get_shape()[1]
        weights = tf.get_variable("weights", [in_dim, out_dim],
                                  dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(0, 0.02))
        biases = tf.Variable(tf.zeros([out_dim]),
                             name='biases')
        activation = tf.matmul(batch_input, weights) + biases
        return activation


def conv(batch_input, out_channels, stride):
    """
    Add convolutional layer to the graph.

    [batch, in_height, in_width, in_channels],
    [filter_width, filter_height, in_channels, out_channels]
    => [batch, out_height, out_width, out_channels]
    """
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4,
                                            in_channels,
                                            out_channels],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))

        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]],
                              mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1],
                            padding="VALID")
        return conv

def deconv(batch_input, out_channels):
    """
    Add transposed convolution to graph.

    [batch, in_height, in_width, in_channels],
    [filter_width, filter_height, out_channels, in_channels]
    => [batch, out_height, out_width, out_channels]
    """
    with tf.variable_scope("deconv"):
        sizes = [int(d) for d in batch_input.get_shape()]
        batch, in_height, in_width, in_channels = sizes
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))

        conv = tf.nn.conv2d_transpose(batch_input, filter,
                                      [batch, in_height * 2, in_width * 2,
                                       out_channels],
                                      [1, 2, 2, 1], padding="SAME")
        return conv

def upsample(batch_input, out_channels, stride):
    """
    Add bilinear upsampling followed by a convolutional layer to the graph.

    [batch, in_height, in_width, in_channels],
    [filter_width, filter_height, in_channels, out_channels]
    => [batch, out_height, out_width, out_channels]
    """
    with tf.variable_scope("upsample"):
        _, in_height, in_width, in_channels = batch_input.get_shape()

        upsampled_input = tf.image.resize_nearest_neighbor(batch_input, [in_height*2, in_width*2]) 

        filter = tf.get_variable("filter", [4, 4,
                                            in_channels,
                                            out_channels],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))

        conv = tf.nn.conv2d(upsampled_input, filter, [1, stride, stride, 1],
                            padding="SAME")
        return conv

def lrelu(x, a):
    """
    Add leaky-relu activation.

    adding these together creates the leak part and linear part
    then cancels them out by subtracting/adding an absolute value term
    leak: a*x/2 - a*abs(x)/2
    linear: x/2 + abs(x)/2
    """
    with tf.name_scope("lrelu"):
        # this block looks like it has 2 inputs
        # on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    """Add bachnorm to layer."""
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer)
        scale = tf.get_variable("scale", [channels],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset,
                                               scale,
                                               variance_epsilon=epsilon)
        return normalized

def instance_norm(input):
    """based conditional_instance_norm from https://github.com/tensorflow/magenta/blob/master/magenta/models/image_stylization/ops.py."""
    with tf.variable_scope("instancenorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer)
        scale = tf.get_variable("scale", [channels],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=False)
        epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset,
                                               scale,
                                               variance_epsilon=epsilon)

    return normalized
