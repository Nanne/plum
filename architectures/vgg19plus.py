"""Couple of layers implemented in tensorflow."""
import tensorflow as tf
import ops

def encoder(encoder_inputs, ngf, instancenorm=False):
    """Return VGG features + extra encoding for image."""

    features = {}
    with tf.variable_scope("block2_conv2"): # 112x112x128
        features['block2_conv2'] = ops.batchnorm(encoder_inputs.pop(0))
    with tf.variable_scope("block3_conv4"): # 56x56x256
        features['block3_conv4'] = ops.batchnorm(encoder_inputs.pop(0))
    with tf.variable_scope("block4_conv4"): # 28x28x512
        features['block4_conv4'] = ops.batchnorm(encoder_inputs.pop(0))
    with tf.variable_scope("block5_conv4"): # 14x14x512
        features['block5_conv4'] = ops.batchnorm(encoder_inputs.pop(0))

    input_layer_spec = 512 # 7x7

    layer_specs = [
        512, # 4x4
        512, # 2x2
        512, # 1x1
    ]

    named_layers, img_embed = ops.encoder(features['block5_conv4'], input_layer_spec, layer_specs, instancenorm)

    features.update(named_layers)

    return features, img_embed

def decoder(encoder_activations, ngf, generator_output_channels,
        drop_prob=0.5, instancenorm=False):
    """Create decoder network, top half of plum-net."""

    layer_specs = [
        (drop_prob, 'encoder_4'),
        (drop_prob, 'encoder_3'),
        (drop_prob, 'encoder_2'),
        (0, 'encoder_1'),
        (0, 'block5_conv4'),
        (0, 'block4_conv4'),
        (0, 'block3_conv4'),
    ]

    output_layer_specs = (generator_output_channels, 0.0, 'block2_conv2')

    return ops.decoder(encoder_activations, layer_specs, output_layer_specs,
            instancenorm=False, upsample_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def discriminator(discrim_inputs, discrim_targets, ndf, instancenorm=False):
    return ops.discriminator(discrim_inputs, discrim_targets, ndf, instancenorm=False)
