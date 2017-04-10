import tensorflow as tf
import glob
import os
import math
from img_ops import transform, img_to_float
import random

FLAGS = tf.app.flags.FLAGS  # parse config

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
            'block2_conv2_size': tf.FixedLenFeature([3], tf.int64),
            'block3_conv4_size': tf.FixedLenFeature([3], tf.int64),
            'block4_conv4_size': tf.FixedLenFeature([3], tf.int64),
            'block5_conv4_size': tf.FixedLenFeature([3], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'image_size': tf.FixedLenFeature([3], tf.int64),
        })
    # Reshape byte-string image to original shape
    path = features['path']
    
    tensors = [path]

    tensors.append(feature_to_shaped(features['block2_conv2'], features['block2_conv2_size'], dtype=tf.float32))
    tensors[-1].set_shape((112, 112, 128))
    tensors.append(feature_to_shaped(features['block3_conv4'], features['block3_conv4_size'], dtype=tf.float32))
    tensors[-1].set_shape((56, 56, 256))
    tensors.append(feature_to_shaped(features['block4_conv4'], features['block4_conv4_size'], dtype=tf.float32))
    tensors[-1].set_shape((28, 28, 512))
    tensors.append(feature_to_shaped(features['block5_conv4'], features['block5_conv4_size'], dtype=tf.float32))
    tensors[-1].set_shape((14, 14, 512))

    # Add target image
    if FLAGS.decoder:
        print 'decoder'
        image = feature_to_shaped(features['image'], features['image_size'], dtype=tf.float64)
        image.set_shape((224,224,3))
        image = tf.cast(image, tf.float32)
        tensors.append(image)
        # Append image again for in summary
        tensors.append(image)

    return tensors

def convert(image, size):
    """Resize image to size if given and convert to unit8."""
    if size:
        image = tf.image.resize_images(image, size=size,
                                       method=tf.image.ResizeMethod.BICUBIC)
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

def preprocess(image):
    """No preprocessing"""
    with tf.name_scope("preprocess"):
        return tf.identity(image)

def deprocess(image):
    """BGR -> RGB, add means"""
    with tf.name_scope("deprocess"):
        means = tf.reshape([103.939, 116.779, 123.68], (1,1,1,3))
        image = tf.add(image, means)
        image = image[:,:,:,::-1]
        return image

