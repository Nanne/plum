import tensorflow as tf
import glob
import os
import math
from img_ops import preprocess, transform, img_to_float
import random

FLAGS = tf.app.flags.FLAGS  # parse config

def feature_to_shaped(feature, shape, dtype=tf.uint8):
    shaped = tf.decode_raw(feature, dtype)

    dimensions = [tf.cast(x, tf.int32) for x in shape]
    dimensions = tf.pack(dimensions)
    shaped = tf.reshape(shaped, dimensions)
    return shaped

def read_record(filename_queue, aux=False):
    """
    Read fromTFrecords containing vgg features.

    based on http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    """
    # Initialize reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Parse TFRecords
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
            'id': tf.FixedLenFeature([], tf.int64),
            'path': tf.FixedLenFeature([], tf.string),
            'block2_conv2': tf.FixedLenFeature([], tf.string),
            'block3_conv4': tf.FixedLenFeature([], tf.string),
            'block4_conv4': tf.FixedLenFeature([], tf.string),
            'block5_conv4': tf.FixedLenFeature([], tf.string),
            'block2_conv2_size': tf.FixedLenFeature([], tf.int64),
            'block3_conv4_size': tf.FixedLenFeature([], tf.int64),
            'block4_conv4_size': tf.FixedLenFeature([], tf.int64),
            'block5_conv4_size': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'image_size': tf.FixedLenFeature([], tf.int64),
        })
    # Reshape byte-string image to original shape
    path = features['path']
    
    vgg_features = []
    vgg_features.append(feature_to_shaped(features['block2_conv2'], features['block2_conv2_size'], dtype=tf.float32))
    vgg_features.append(feature_to_shaped(features['block3_conv4'], features['block3_conv4_size'], dtype=tf.float32))
    vgg_features.append(feature_to_shaped(features['block4_conv4'], features['block4_conv4_size'], dtype=tf.float32))
    vgg_features.append(feature_to_shaped(features['block5_conv4'], features['block5_conv4_size'], dtype=tf.float32))

    tensors = [path, vgg_features]

    # Add target image
    if FLAGS.decoder:
        print 'decoder'
        image = feature_to_shaped(features['image'], features['image_size'])
        tensors.append(image)

    return tensors
