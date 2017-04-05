
"""Couple of layers implemented in tensorflow."""
import tensorflow as tf
import ops

def encoder_unet(encoder_inputs, ngf, instancenorm=False):
    """Create image encoder. Bottom half of U-net."""

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    input_layer_spec = ngf

    # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
    # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
    # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
    # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
    # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
    # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
    # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    layer_specs = [
        ngf * 2,
        ngf * 4,
        ngf * 8,
        ngf * 8,
        ngf * 8,
        ngf * 8,
        ngf * 8,
    ]

    return encoder(encoder_inputs, input_layer_spec, layer_specs, instancenorm)

def decoder_unet(encoder_activations, ngf, generator_outputs_channels,
        drop_prob=0.5, instancenorm=False):
    """Create decoder network, top half of U-net."""
    # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
    # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
    # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
    # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
    # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
    # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
    # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]

    layer_specs = [
        (drop_prob, 'encoder_8'),
        (drop_prob, 'encoder_7'),
        (drop_prob, 'encoder_6'),
        (0.0, 'encoder_5'),
        (0.0, 'encoder_4'),
        (0.0, 'encoder_3'),
        (0.0, 'encoder_2')
    ]

    output_layer_spec = (generator_output_channels, 0.0, 'encoder_1')

    return decoder(encoder_activations, layer_spec, output_layer_specs,
            drop_prob=0.5, instancenorm=False)


def decoder_vgg19(encoder_activations, generator_outputs_channels,
        drop_prob=0.5, instancenorm=False):
    """Create decoder network, top half of plum-net."""
    # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 14, 14, ngf * 8 * 2]
    # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 28, 28, ngf * 4 * 2]
    # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 56, 56, ngf * 2 * 2]
    # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 112, 112, ngf * 2]

    layer_specs = [
        (drop_prob, 'block5_conv2'),
        (drop_prob, 'block4_conv4'),
        (drop_prob, 'block3_conv4'),
    ]

    output_layer_spec = (generator_output_channels, 0.0, 'block2_conv2')

    return decoder(encoder_activations, layer_spec, output_layer_specs,
            drop_prob=0.5, instancenorm=False)

def encoder(encoder_inputs, input_layer_spec, layer_specs, instancenorm=False):
    """Create image encoder. Based on layer specs."""
    layers = []
    named_layers = {}

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, input_layer_spec]
    with tf.variable_scope("encoder_1"):
        output = ops.conv(encoder_inputs, input_layer_spec, stride=2)
        layers.append(output)
        named_layers["encoder_1"] = layers[-1]

    norm = ops.instance_norm if instancenorm else ops.batchnorm
    for out_channels in layer_specs:
        scope_name = "encoder_%d" % (len(layers) + 1)
        with tf.variable_scope(scope_name):
            rectified = ops.lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels]
            # => [batch, in_height/2, in_width/2, out_channels]
            convolved = ops.conv(rectified, out_channels, stride=2)
            output = norm(convolved)
            layers.append(output)
            named_layers[scope_name] = layers[-1]

    img_embed = tf.squeeze(layers[-1], [1, 2])
    return named_layers, img_embed

def decoder(input_layers, layer_specs, output_layer_spec,
            drop_prob=0.5, instancenorm=False):
    """Create decoder network  based on some layerspec."""

    layers = []
    named_layers = {}

    norm = ops.instance_norm if instancenorm else ops.batchnorm
    num_encoder_layers = len(input_layers)
    for decoder_layer, (dropout, skip_layer) in enumerate(layer_specs):
        # Number of out channels is equal to the number of channels of
        # the skip connection we'll be concat with
        next_skip = layer_spec[decoder_layer+1][1]
        out_channels = input_layers[next_skip].get_shape()[3]

        scope_name = "decoder_%d" % (len(layer_spec) + 1 - decoder_layer)
        with tf.variable_scope(scope_name):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = input_layers[skip_layer]
            else:
                input = tf.concat(values=[layers[-1], input_layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels]
            # => [batch, in_height*2, in_width*2, out_channels]
            output = ops.upsample(rectified, out_channels)
            output = norm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)
            named_layers[scope_name] = layers[-1]

    # decoder_1: [batch, 128, 128, ngf * 2]
    # => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        out_channels, dropout, skip_layer = output_layer_spec

        input = tf.concat(values=[layers[-1], input_layers[skip_layer]], axis=3)
        rectified = tf.nn.relu(input)
        output = ops.upsample(rectified, out_channels)
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
    input = tf.concat_v2([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = ops.conv(input, ndf, stride=2)
        rectified = ops.lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    norm = ops.instance_norm if instancenorm else ops.batchnorm
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            # last layer here has stride 1
            stride = 1 if i == n_layers - 1 else 2
            convolved = ops.conv(layers[-1], out_channels, stride=stride)
            normalized = norm(convolved)
            rectified = ops.lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = ops.conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]
