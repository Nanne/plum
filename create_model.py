from util import to_namedtuple
import models.ops as ops
import tensorflow as tf

EPS = 1e-12
FLAGS = tf.app.flags.FLAGS  # parse config

if FLAGS.architecture == "unet":
    from models.unet import encoder, decoder, discriminator
elif FLAGS.architecture == "vgg19":
    from models.vgg19 import encoder, decoder, discriminator
elif FLAGS.architecture == "vgg19plus":
    from models.vgg19plus import encoder, decoder, discriminator
else:
    raise ValueError('Unknown architecture option')

def create_model(e):
    """Create U-net with optional discriminator and auxiliary classifier."""
    m = {}
    with tf.variable_scope("generator") as scope:
        # Image encoder
        with tf.name_scope("encoder"):
            encoder_activations, image_embedding = encoder(e.inputs,
                                                           FLAGS.ngf)
        # Image decoder
        if e.targets != None:
            with tf.name_scope("decoder"):
                out_channels = int(e.targets.get_shape()[-1])
                if FLAGS.mode == "test":
                    outputs = decoder(encoder_activations, FLAGS.ngf,
                                      out_channels, drop_prob=0.0)
                else:
                    outputs = decoder(encoder_activations, FLAGS.ngf,
                                      out_channels, drop_prob=0.5)
                m['outputs'] = outputs
        # Classifier on top of the encoder
        if e.aux != None:
            with tf.name_scope("classifier"):
                logits = ops.dense(image_embedding, FLAGS.num_classes)
    # Add discriminator and GAN loss
    if e.discriminator_inputs != None:
        # create two copies of discriminator,
        # one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels]
                # => [batch, 30, 30, 1]
                    predict_real = discriminator(e.discriminator_inputs,
                                                 e.targets, FLAGS.ndf)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = discriminator(e.discriminator_inputs,
                                             outputs, FLAGS.ndf)

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
        gen_loss_content = 0
        if e.targets != None:
            if FLAGS.bat_loss > 0:
                logits_pred = tf.reshape(outputs, [FLAGS.batch_size, -1])
                target_flat = tf.reshape(e.targets, [FLAGS.batch_size, -1])
                prob_pred = tf.nn.softmax(logits_pred)
                prob_target = tf.nn.softmax(target_flat)
                gen_loss_bat = -tf.log(tf.reduce_sum(tf.sqrt(tf.multiply(prob_pred,
                                                             prob_target))))
                gen_loss_content += FLAGS.bat_loss * gen_loss_bat
            if FLAGS.l1_loss > 0:
                gen_loss_L1 = tf.reduce_mean(tf.abs(e.targets - outputs))
                gen_loss_content += FLAGS.l1_loss * gen_loss_L1
            if FLAGS.l2_loss > 0:
                gen_loss_L2 = tf.reduce_mean(tf.pow(e.targets - outputs,2))
                gen_loss_content += FLAGS.l2_loss * gen_loss_L2

            gen_loss += FLAGS.content_weight * gen_loss_content
        if e.discriminator_inputs != None:
            with tf.name_scope("generator_gan_loss"):
                gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
                gen_loss += gen_loss_GAN * FLAGS.gan_weight
        if e.aux != None:
            with tf.name_scope("classification_loss"):
                aux_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits,
                                                                   e.aux)
                aux_loss = tf.reduce_mean(aux_loss)
                gen_loss += FLAGS.aux_weight * aux_loss

    if e.discriminator_inputs != None:
        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(FLAGS.lr, FLAGS.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
            m['discrim_grads_and_vars'] = discrim_grads_and_vars

    with tf.name_scope("generator_train"):
        if e.discriminator_inputs != None:
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
    if e.targets != None:
        losses.append(gen_loss_content)
    if e.discriminator_inputs != None:
        losses.append(discrim_loss)
        losses.append(gen_loss_GAN)
    if e.aux != None:
        losses.append(aux_loss)
    update_losses = ema.apply(losses)
    if e.targets != None:
        m['gen_loss_content'] = ema.average(gen_loss_content)
    if e.discriminator_inputs != None:
        m['discrim_loss'] = ema.average(discrim_loss)
        m['gen_loss_GAN'] = ema.average(gen_loss_GAN)
    if e.aux != None:
        m['aux_loss'] = ema.average(aux_loss)

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)
    m['train'] = tf.group(update_losses, incr_global_step, gen_train)
    model = to_namedtuple(m, "Model")
    return model
