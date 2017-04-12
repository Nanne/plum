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
            'fc1': tf.FixedLenFeature([], tf.string),
            'fc1_size': tf.FixedLenFeature([4], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'image_size': tf.FixedLenFeature([3], tf.int64),
        })

    path = features['path']
    tensors = [path]

    # Reshape byte-strings to original shape
    tensors.append(feature_to_shaped(features['fc1'], features['fc1_size'], dtype=tf.float32))
    tensors[-1].set_shape((4096))

    image = feature_to_shaped(features['image'], features['image_size'], dtype=tf.uint8)
    image.set_shape((224,224,3))

    tensors.append(preprocess_output(image))
    # Append image again for in summary
    tensors.append(preprocess_input(image))

    return tensors

def preprocess_input(image):
    """ uint8 [0,255] -> float32 [-1, 1] """
    with tf.name_scope("preprocess_input"):
        image = tf.image.convert_image_dtype(image, tf.float32, saturate=True)
        return image * 2 - 1

def deprocess_input(image):
    """float32 [-1, 1] => uint8 [0, 255]."""
    with tf.name_scope("deprocess_input"):
        image = (image + 1) / 2
        return tf.image.convert_image_dtype(image, tf.uint8, saturate=True)

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
