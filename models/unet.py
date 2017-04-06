"""Couple of layers implemented in tensorflow."""
import tensorflow as tf
import ops

def encoder(encoder_inputs, ngf, instancenorm=False):
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

    return ops.encoder(encoder_inputs, input_layer_spec, layer_specs, instancenorm)

def decoder(encoder_activations, ngf, generator_outputs_channels,
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

    return ops.decoder(encoder_activations, layer_spec, output_layer_specs,
            drop_prob=0.5, instancenorm=False)

def discriminator(discrim_inputs, discrim_targets, ndf, instancenorm=False):
    return ops.discriminator(discrim_inputs, discrim_targets, ndf, instancenorm=False)
