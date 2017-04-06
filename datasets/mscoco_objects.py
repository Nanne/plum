import tensorflow as tf
import glob
import os
import math
from img_ops import preprocess, transform, img_to_float
import random

FLAGS = tf.app.flags.FLAGS  # parse config


def read_record(filename_queue, aux=False):
    """
    Read .

    based on http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    """
    flip = FLAGS.mode == "train"
    # Initialize reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Parse TFRecords
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
            'cocoid': tf.FixedLenFeature([], tf.int64),
            'path': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'objects': tf.FixedLenFeature([], tf.string),
        })
    # Reshape byte-string image to original shape
    path = features['path']
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image_shape = tf.pack([height, width, depth])
    image = tf.reshape(image, image_shape)
    image = preprocess(img_to_float(image))
    image = transform(image, flip, FLAGS.seed,
                                 FLAGS.scale_size, FLAGS.crop_size)
    if aux:
        objs = tf.decode_raw(features['objects'], tf.int64)
        objs = tf.cast(objs, tf.float32)
        objs.set_shape(FLAGS.num_classes)
        # print objs
        # objs = tf.reshape(objs, (tf.cast(90, tf.int32), ))
        return path, image, objs

    return path, image
