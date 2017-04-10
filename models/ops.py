
"""Couple of layers implemented in tensorflow."""
import tensorflow as tf

if tf.__version__ == "0.12.1":
    concatenate = tf.concat_v2
else:
    concatenate = tf.concat

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
        sizes = [int(d) for d in batch_input.get_shape()]
        _, in_height, in_width, in_channels = sizes

        upsampled_input = tf.image.resize_nearest_neighbor(batch_input,
                                                          [int(in_height)*2,
                                                           int(in_width)*2])

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
                                 initializer=tf.zeros_initializer())
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

def encoder(encoder_inputs, input_layer_spec, layer_specs, instancenorm=False):
    """Create image encoder. Based on layer specs."""
    layers = []
    named_layers = {}

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, input_layer_spec]
    with tf.variable_scope("encoder_1"):
        output = conv(encoder_inputs, input_layer_spec, stride=2)
        layers.append(output)
        named_layers["encoder_1"] = layers[-1]

    norm = instance_norm if instancenorm else batchnorm
    for out_channels in layer_specs:
        scope_name = "encoder_%d" % (len(layers) + 1)
        with tf.variable_scope(scope_name):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels]
            # => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = norm(convolved)
            layers.append(output)
            named_layers[scope_name] = layers[-1]

    img_embed = tf.squeeze(layers[-1], [1, 2])
    return named_layers, img_embed

def decoder(input_layers, layer_specs, output_layer_specs,
            drop_prob=0.5, instancenorm=False, upsample=False):
    """Create decoder network  based on some layerspec."""

    layers = []
    named_layers = {}

    norm = instance_norm if instancenorm else batchnorm
    num_encoder_layers = len(input_layers)
    for decoder_layer, (dropout, skip_layer) in enumerate(layer_specs):
        # Number of out channels is equal to the number of channels of
        # the skip connection we'll be concat with
        if decoder_layer < len(layer_specs) - 1:
            next_skip = layer_specs[decoder_layer+1][1]
        else:
            next_skip = output_layer_specs[2]
        out_channels = int(input_layers[next_skip].get_shape()[3])

        scope_name = "decoder_%d" % (len(layer_specs) + 1 - decoder_layer)
        with tf.variable_scope(scope_name):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = input_layers[skip_layer]
            else:
                input = concatenate(values=[layers[-1], input_layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels]
            # => [batch, in_height*2, in_width*2, out_channels]
            if upsample:
                output = upsample(rectified, out_channels, stride=2)
            else:
                output = deconv(rectified, out_channels)
            output = norm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)
            named_layers[scope_name] = layers[-1]

    # decoder_1: [batch, 128, 128, ngf * 2]
    # => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        out_channels, dropout, skip_layer = output_layer_specs

        input = concatenate(values=[layers[-1], input_layers[skip_layer]], axis=3)
        rectified = tf.nn.relu(input)
        if upsample:
            output = upsample(rectified, out_channels, stride=2)
        else:
            output = deconv(rectified, out_channels)
        output = tf.tanh(output)

        if dropout > 0.0:
            output = tf.nn.dropout(output, keep_prob=1 - dropout)

        layers.append(output)
        named_layers["decoder_1"] = layers[-1]

    return layers[-1]


def discriminator(discrim_inputs, discrim_targets, ndf, instancenorm=False):
    """Create discriminator network."""
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels]
    # => [batch, height, width, in_channels * 2]
    input = concatenate(values=[discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    norm = instance_norm if instancenorm else batchnorm
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            # last layer here has stride 1
            stride = 1 if i == n_layers - 1 else 2
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = norm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]
