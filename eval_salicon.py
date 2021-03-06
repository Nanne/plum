from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import json
import os

import time
import ConfigParser
import sys
sys.path.append("/roaming/public_datasets/SALICON/PythonAPI/")
sys.path.append("/roaming/public_datasets/SALICON/salicon-evaluation/")
from image2json import ImageTools
from salicon.salicon import SALICON
from saliconeval.eval import SALICONEval

from create_model import create_model
from dataprovider import load_records
import cfg, util, dataprovider
import sys

CROP_SIZE = 256

FLAGS = tf.app.flags.FLAGS  # parse config
FLAGS.crop_size = CROP_SIZE
FLAGS.mode = "test"

'''
config = ConfigParser.ConfigParser()
config.readfp(open(r'salicon.cfg'))
if FLAGS.records:
    FLAGS.input_dir = config.get("Paths", "val_record")
else:
    FLAGS.input_dir = config.get("Paths", "val_path")
FLAGS.scale_size = config.getint("Image", "scale_size")
FLAGS.upsample_w = config.getint("Image", "upsample_w")
FLAGS.upsample_h = config.getint("Image", "upsample_h")
FLAGS.which_direction = config.get("Image", "which_direction")
'''
annFile = "/roaming/public_datasets/SALICON/annotations/fixations_val2015r1.json"
resFile = os.path.join(FLAGS.output_dir, "fullval_result.json")
salicon = SALICON(annFile)


def run_eval(checkpoint, saver, examples, display_fetches):
    """Load model, predict saliency maps, use prediciton with SALICON api."""
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        print("loading model from checkpoint:" + checkpoint)
        saver.restore(sess, checkpoint)

        max_steps = examples.steps_per_epoch
        for step in range(max_steps):
            results = sess.run(display_fetches)
            filesets = util.save_images(results, FLAGS.output_dir, only_output=True)
            for i, f in enumerate(filesets):
                print("evaluated image", f["name"])
            index_path = util.append_index(filesets, FLAGS.output_dir)

    print("wrote index at", index_path)
    result_str = ""
    sp = ImageTools(FLAGS.output_dir+"/images/", resFile)
    sp.convert()
    sp.dumpRes()

    # initialize COCO ground truth and results api's
    saliconRes = salicon.loadRes(resFile)

    # create cocoEval object by taking coco and cocoRes
    saliconEval = SALICONEval(salicon, saliconRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    saliconEval.params['image_id'] = saliconRes.getImgIds()

    # evaluate results
    saliconEval.evaluate()

    print("Final Result for each Metric:")
    for metric, score in saliconEval.eval.items():
        print('%s: %.3f' % (metric, score))
        result_str += '%s: %.3f' % (metric, score) + '\n'
    result_str += '\n\n'

    with open("results.txt", "a") as f:
        f.write(checkpoint + '\n\n')
        f.write(result_str)

    return checkpoint


def main():
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    #
    util.restore_flags()
    # disable these features in test mode
    FLAGS.flip = False

    for k, v in FLAGS.__dict__['__flags'].items():
        print(k, "=", v)

    with open(os.path.join(FLAGS.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(FLAGS.__dict__['__flags'],
                           sort_keys=True, indent=4))

    print(FLAGS.aux)
    examples = load_records()

    deprocess_input = examples.deprocess_input
    deprocess_output = examples.deprocess_output

    print("examples count = %d" % examples.count)
    model = create_model(examples)


    # summaries
    with tf.name_scope("images_summary"):
        deprocessed_images = deprocess_input(examples.images)
        tf.summary.image("images", deprocessed_images)

    if FLAGS.decoder:
        with tf.name_scope("targets_summary"):
            deprocessed_targets = deprocess_output(examples.targets)
            tf.summary.image("targets", deprocessed_targets)

        with tf.name_scope("outputs_summary"):
            deprocessed_outputs = deprocess_output(model.outputs)
            tf.summary.image("outputs", deprocessed_outputs)

        with tf.name_scope("encode_images"):
            display_fetches = {
                "paths": examples.paths,
                "images": tf.map_fn(tf.image.encode_png, deprocessed_images,
                                    dtype=tf.string, name="input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, deprocessed_targets,
                                     dtype=tf.string, name="target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png, deprocessed_outputs,
                                     dtype=tf.string, name="output_pngs"),
            }

    # reverse any processing on images so they can
    # be written to disk or displayed to user
    saver = tf.train.Saver()
    old_checkpoint = None
    while True:
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
        if checkpoint != old_checkpoint:
            old_checkpoint = run_eval(checkpoint, saver, examples,
                                      display_fetches)
        # Sleep 1 minute
        time.sleep(60 * 1)

main()
