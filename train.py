from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import json
import random
import math
import time
import cfg, util, dataprovider
from create_model import create_model

FLAGS = tf.app.flags.FLAGS  # parse config
CROP_SIZE = 256
FLAGS.crop_size = CROP_SIZE

def main(_):
    """Run Everything."""
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    if FLAGS.salicon:
        util.set_salicon()
    
    if FLAGS.output_dir != None:
        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
    else:
        raise Exception("output_dir required")

    if FLAGS.mode == "export":
        if FLAGS.checkpoint is None:
            raise Exception("checkpoint required for export mode")

        # load some options from the checkpoint
        util.restore_flags()
        # disable these features in test mode
        FLAGS.scale_size = CROP_SIZE
        FLAGS.flip = False
    else:
        for k, v in FLAGS.__dict__['__flags'].items():
            print(k, "=", v)

        with open(os.path.join(FLAGS.output_dir, "options.json"), "w") as f:
            f.write(json.dumps(FLAGS.__dict__['__flags'],
                               sort_keys=True, indent=4))

    examples = dataprovider.load_records()

    # Retrieve data specific functions
    deprocess_input = examples.deprocess_input
    deprocess_output = examples.deprocess_output

    print("examples count = %d" % examples.count)
    if FLAGS.decoder and FLAGS.aux:
        model = create_model(examples.inputs, examples.targets, examples.aux)
    elif FLAGS.aux:
        model = create_model(examples.inputs, None, examples.aux)
    elif FLAGS.decoder:
        model = create_model(examples.inputs, examples.targets)
    else:
        raise Exception("At least on of --aux or --decoder has to be True")

    # summaries
    with tf.name_scope("images_summary"):
        converted_images = deprocess_input(examples.images)
        tf.summary.image("images", converted_images)

    if FLAGS.decoder:
        with tf.name_scope("targets_summary"):
            converted_targets = deprocess_output(examples.targets)
            tf.summary.image("targets", converted_targets)

        with tf.name_scope("outputs_summary"):
            converted_outputs = deprocess_output(model.outputs)
            tf.summary.image("outputs", converted_outputs)

        with tf.name_scope("encode_images"):
            display_fetches = {
                "paths": examples.paths,
                "images": tf.map_fn(tf.image.encode_png, converted_images,
                                    dtype=tf.string, name="input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, converted_targets,
                                     dtype=tf.string, name="target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png, converted_outputs,
                                     dtype=tf.string, name="output_pngs"),
            }


    if FLAGS.decoder:
        tf.summary.scalar("generator_loss_content", model.gen_loss_content)

    if FLAGS.discriminator:
        with tf.name_scope("predict_real_summary"):
            tf.summary.image("predict_real",
                             tf.image.convert_image_dtype(model.predict_real,
                                                          dtype=tf.uint8))
        with tf.name_scope("predict_fake_summary"):
            tf.summary.image("predict_fake",
                             tf.image.convert_image_dtype(model.predict_fake,
                                                          dtype=tf.uint8))
        tf.summary.scalar("discriminator_loss", model.discrim_loss)
        tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)

    if FLAGS.aux:
        tf.summary.scalar("aux_loss", model.aux_loss)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    if FLAGS.discriminator:
        g_a_v = model.discrim_grads_and_vars + model.gen_grads_and_vars
    else:
        g_a_v = model.gen_grads_and_vars
    for grad, var in g_a_v:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = FLAGS.output_dir if (FLAGS.trace_freq > 0 or FLAGS.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if FLAGS.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if FLAGS.max_epochs > 0:
            max_steps = examples.steps_per_epoch * FLAGS.max_epochs
        if FLAGS.mode == "train":
            # training
            start_time = time.time()
            print("START TRAIN!!!!")
            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(FLAGS.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }
                if should(FLAGS.progress_freq):
                    if FLAGS.decoder:
                        fetches["gen_loss_content"] = model.gen_loss_content
                    if FLAGS.discriminator:
                        fetches["discrim_loss"] = model.discrim_loss
                        fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    if FLAGS.aux:
                        fetches["aux_loss"] = model.aux_loss

                if should(FLAGS.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(FLAGS.display_freq) and FLAGS.decoder:
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options,
                                   run_metadata=run_metadata)

                global_step = results["global_step"]

                if should(FLAGS.summary_freq):
                    sv.summary_writer.add_summary(results["summary"],
                                                  global_step)

                if should(FLAGS.display_freq):
                    print("saving display images")
                    filesets = util.save_images(results["display"],
                                           step=global_step)
                    util.append_index(filesets, step=True)

                if should(FLAGS.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata,
                                                       "step_%d" % global_step)

                if should(FLAGS.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(global_step / examples.steps_per_epoch)
                    train_step = global_step - (train_epoch - 1) * examples.steps_per_epoch
                    print("progress  epoch %d  step %d  image/sec %0.1f" % (train_epoch, train_step, global_step * FLAGS.batch_size / (time.time() - start_time)))
                    if FLAGS.decoder:
                        print("gen_loss_content", results["gen_loss_content"])
                    if FLAGS.discriminator:
                        print("discrim_loss", results["discrim_loss"])
                        print("gen_loss_GAN", results["gen_loss_GAN"])
                    if FLAGS.aux:
                        print("aux_loss", results["aux_loss"])

                if should(FLAGS.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(FLAGS.output_dir, "model"),
                               global_step=sv.global_step)

                if sv.should_stop():
                    break

if __name__ == '__main__':
    tf.app.run()
