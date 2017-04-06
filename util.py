import json
import tensorflow as tf
import numpy as np
import os.path
from collections import namedtuple

import cfg
FLAGS = tf.app.flags.FLAGS

def restore_flags():
    if tf.gfile.Exists(os.path.join(tf.app.flags.FLAGS.checkpoint, 'options.json')):
        with open(os.path.join(tf.app.flags.FLAGS.checkpoint, 'options.json'), 'r') as f:
            print('Restoring training flags')
            train_flags = json.load(f)

            for key in cfg.restore_flags:
                if key in train_flags:
                    tf.app.flags.FLAGS.__dict__['__flags'][key] = train_flags[key]
                print(key, tf.app.flags.FLAGS.__dict__['__flags'][key])
    else:
        print('No flag configuration file found, using default flags')
    return

def set_salicon():
    """Set FLAGS from SALICON data set from 'salicon.cfg'."""
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.readfp(open(r'salicon.cfg'))

    if FLAGS.mode == "train":
        if FLAGS.records:
            FLAGS.input_dir = config.get("Paths", "train_record")
        else:
            FLAGS.input_dir = config.get("Paths", "train_path")
    else:
        if FLAGS.records:
            FLAGS.input_dir = config.get("Paths", "val_record")
        else:
            FLAGS.input_dir = config.get("Paths", "val_path")
    FLAGS.scale_size = config.getint("Image", "scale_size")
    FLAGS.upsample_w = config.getint("Image", "upsample_w")
    FLAGS.upsample_h = config.getint("Image", "upsample_h")
    FLAGS.which_direction = config.get("Image", "which_direction")

def to_namedtuple(dictionary, type_name):
    """Convert dictionary into named tuple."""
    return namedtuple(type_name, dictionary.keys())(**dictionary)

def get_name(path):
    """Get filename given path."""
    name, _ = os.path.splitext(os.path.basename(path))
    return name

def save_images(fetches, output_dir, step=None, only_output=False):
    """Write input, target and output images to output_dir."""
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            if only_output and kind == "outputs":
                filename = name + ".png"
            else:
                filename = name + "-" + kind + ".png"

            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            if only_output and kind != "outputs":
                pass
            else:
                with open(out_path, "wb") as f:
                    f.write(contents)
        filesets.append(fileset)
    return filesets

def append_index(filesets, output_dir, step=False):
    """Create html output of the results."""
    index_path = os.path.join(output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path
