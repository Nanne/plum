"""Couple of layers implemented in tensorflow."""
import tensorflow as tf
import ops

def encoder(encoder_inputs, ngf, instancenorm=False):
    """Return VGG features for image."""

    features = {}
    with tf.variable_scope("block2_conv2"):
        features['block2_conv2'] = ops.batchnorm(encoder_inputs.pop(0))
    with tf.variable_scope("block3_conv4"):
        features['block3_conv4'] = ops.batchnorm(encoder_inputs.pop(0))
    with tf.variable_scope("block4_conv4"):
        features['block4_conv4'] = ops.batchnorm(encoder_inputs.pop(0))
    with tf.variable_scope("block5_conv4"):
        features['block5_conv4'] = ops.batchnorm(encoder_inputs.pop(0))

    return features, None

def decoder(encoder_activations, ngf, generator_output_channels,
        drop_prob=0.5, instancenorm=False):
    """Create decoder network, top half of plum-net."""
    # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 14, 14, ngf * 8 * 2]
    # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 28, 28, ngf * 4 * 2]
    # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 56, 56, ngf * 2 * 2]
    # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 112, 112, ngf * 2]

    layer_specs = [
        (drop_prob, 'block5_conv4'),
        (drop_prob, 'block4_conv4'),
        (drop_prob, 'block3_conv4'),
    ]

    output_layer_specs = (generator_output_channels, 0.0, 'block2_conv2')

    return ops.decoder(encoder_activations, layer_specs, output_layer_specs,
            instancenorm=False, do_upsample=True)

def discriminator(discrim_inputs, discrim_targets, ndf, instancenorm=False):
    return ops.discriminator(discrim_inputs, discrim_targets, ndf, instancenorm=False)
