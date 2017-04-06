from util import to_namedtuple
import models.ops as ops
import tensorflow as tf

EPS = 1e-12
FLAGS = tf.app.flags.FLAGS  # parse config

if FLAGS.architecture == "unet":
    from models.unet import encoder, decoder, discriminator
elif FLAGS.architecture == "vgg19":
    from models.vgg19 import encoder, decoder, discriminator
else:
    raise ValueError('Unknown architecture option')

def create_model(inputs, targets, aux_targets=None):
    """Create U-net with optional discriminator and auxiliary classifier."""
    m = {}
    with tf.variable_scope("generator") as scope:
        # Image encoder
        with tf.name_scope("encoder"):
            encoder_activations, image_embedding = encoder(inputs, FLAGS.ngf)
        # Image decoder
        if FLAGS.decoder:
            with tf.name_scope("decoder"):
                out_channels = int(targets.get_shape()[-1])
                if FLAGS.mode == "test":
                    outputs = decoder(encoder_activations, FLAGS.ngf,
                                      out_channels, drop_prob=0.0)
                else:
                    outputs = decoder(encoder_activations, FLAGS.ngf,
                                      out_channels, drop_prob=0.5)
                m['outputs'] = outputs
        # Classifier on top of the encoder
        if FLAGS.aux:
            with tf.name_scope("classifier"):
                logits = ops.dense(image_embedding, FLAGS.num_classes)
    # Add discriminator and GAN loss
    if FLAGS.discriminator:
        # create two copies of discriminator,
        # one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels]
                # => [batch, 30, 30, 1]
                predict_real = discriminator(inputs, targets, FLAGS.ndf)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = discriminator(inputs, outputs, FLAGS.ndf)

        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        m['predict_real'] = predict_real
        m['predict_fake'] = predict_fake
        m['discriminator_loss'] = discrim_loss

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        # Compute softmax on target and output image
        # Compute Bhataccarya distance
        gen_loss = 0
        if FLAGS.decoder:
            logits_pred = tf.reshape(outputs, [FLAGS.batch_size, -1])
            target_flat = tf.reshape(targets, [FLAGS.batch_size, -1])
            prob_pred = tf.nn.softmax(logits_pred)
            prob_target = tf.nn.softmax(target_flat)
            gen_loss_bat = - tf.log(tf.reduce_sum(tf.sqrt(tf.multiply(prob_pred, prob_target))))
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            if FLAGS.content_loss == 'bat':
                gen_loss_content = gen_loss_bat
            elif FLAGS.content_loss == 'L1':
                gen_loss_content = gen_loss_L1
            elif FLAGS.content_loss == 'both':
                gen_loss_content = 0.1 * gen_loss_L1 + 0.9 * gen_loss_bat
            else:
                raise NotImplementedError("Loss {} not implemented".format(FLAGS.content_loss))
            gen_loss += FLAGS.content_weight * gen_loss_content
        if FLAGS.discriminator:
            with tf.name_scope("generator_gan_loss"):
                gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
                gen_loss += gen_loss_GAN * FLAGS.gan_weight
        if FLAGS.aux:
            with tf.name_scope("classification_loss"):
                aux_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits,
                                                                   aux_targets)
                aux_loss = tf.reduce_mean(aux_loss)
                gen_loss += FLAGS.aux_weight * aux_loss

    if FLAGS.discriminator:
        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(FLAGS.lr, FLAGS.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
            m['discrim_grads_and_vars'] = discrim_grads_and_vars

    with tf.name_scope("generator_train"):
        if FLAGS.discriminator:
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(FLAGS.lr, FLAGS.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
                m['gen_grads_and_vars'] = gen_grads_and_vars
        else:
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(FLAGS.lr, FLAGS.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss,
                                                             var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
            m['gen_grads_and_vars'] = gen_grads_and_vars

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    losses = []
    if FLAGS.decoder:
        losses.append(gen_loss_content)
    if FLAGS.discriminator:
        losses.append(discrim_loss)
        losses.append(gen_loss_GAN)
    if FLAGS.aux:
        losses.append(aux_loss)
    update_losses = ema.apply(losses)
    if FLAGS.decoder:
        m['gen_loss_content'] = ema.average(gen_loss_content)
    if FLAGS.discriminator:
        m['discrim_loss'] = ema.average(discrim_loss)
        m['gen_loss_GAN'] = ema.average(gen_loss_GAN)
    if FLAGS.aux:
        m['aux_loss'] = ema.average(aux_loss)

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)
    m['train'] = tf.group(update_losses, incr_global_step, gen_train)
    model = to_namedtuple(m, "Model")
    return model
