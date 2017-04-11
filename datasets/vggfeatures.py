import tensorflow as tf
import glob
import os
import math
from img_ops import transform, img_to_float
import random

FLAGS = tf.app.flags.FLAGS  # parse config
FLAGS.pretrained = True

def feature_to_shaped(feature, shape, dtype=tf.uint8):
    shaped = tf.decode_raw(feature, dtype)
    shaped = tf.squeeze(tf.reshape(shaped, tf.cast(shape, tf.int32)))
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
            'id': tf.FixedLenFeature([1], tf.int64),
            'path': tf.FixedLenFeature([], tf.string),
            'block2_conv2': tf.FixedLenFeature([], tf.string),
            'block3_conv4': tf.FixedLenFeature([], tf.string),
            'block4_conv4': tf.FixedLenFeature([], tf.string),
            'block5_conv4': tf.FixedLenFeature([], tf.string),
            'block2_conv2_size': tf.FixedLenFeature([4], tf.int64),
            'block3_conv4_size': tf.FixedLenFeature([4], tf.int64),
            'block4_conv4_size': tf.FixedLenFeature([4], tf.int64),
            'block5_conv4_size': tf.FixedLenFeature([4], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'image_size': tf.FixedLenFeature([3], tf.int64),
        })

    path = features['path']
    tensors = [path]

    # Reshape byte-strings to original shape
    tensors.append(feature_to_shaped(features['block2_conv2'], features['block2_conv2_size'], dtype=tf.float32))
    tensors[-1].set_shape((112, 112, 128))
    tensors.append(feature_to_shaped(features['block3_conv4'], features['block3_conv4_size'], dtype=tf.float32))
    tensors[-1].set_shape((56, 56, 256))
    tensors.append(feature_to_shaped(features['block4_conv4'], features['block4_conv4_size'], dtype=tf.float32))
    tensors[-1].set_shape((28, 28, 512))
    tensors.append(feature_to_shaped(features['block5_conv4'], features['block5_conv4_size'], dtype=tf.float32))
    tensors[-1].set_shape((14, 14, 512))

    image = feature_to_shaped(features['image'], features['image_size'], dtype=tf.uint8)
    image.set_shape((224,224,3))

    tensors.append(preprocess_output(image))
    # Append image again for in summary
    tensors.append(image)

    return tensors

def preprocess_input(image):
    """Identity."""
    with tf.name_scope("preprocess_input"):
        return tf.identity(image)

def deprocess_input(image):
    """Identity."""
    with tf.name_scope("deprocess_input"):
        return tf.identity(image)

def preprocess_output(image):
    """ uint8 [0,255] -> float32 [-1, 1] """
    with tf.name_scope("preprocess_output"):
        image = tf.image.convert_image_dtype(image, tf.float32, saturate=True)
        return image * 2 - 1

def deprocess_output(image):
    """float32 [-1, 1] => uint8 [0, 255]."""
    with tf.name_scope("deprocess_output"):
        image = (image + 1) / 2
        return tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
