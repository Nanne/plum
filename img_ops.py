"""Tensorflow image operations."""
from __future__ import division
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS  # parse config


def preprocess(image):
    """[0, 1] => [-1, 1]."""
    with tf.name_scope("preprocess"):
        return image * 2 - 1


def deprocess(image):
    """[-1, 1] => [0, 1]."""
    with tf.name_scope("deprocess"):
        return (image + 1) / 2


def img_to_float(image):
    """Convert image to float."""
    raw_input = tf.image.convert_image_dtype(image, dtype=tf.float32)
    assertion = tf.assert_equal(tf.shape(raw_input)[2], 3,
                                message="image does not have 3 channels")
    with tf.control_dependencies([assertion]):
        raw_input = tf.identity(raw_input)
    raw_input.set_shape([None, None, 3])
    return raw_input

def transform(image, flip, seed, scale_size, crop_size):
    """Flip and resize images."""
    r = image
    if flip:
        r = tf.image.random_flip_left_right(r, seed=seed)

    # area produces a nice downscaling,
    # but does nearest neighbor for upscaling
    # assume we're going to be doing downscaling here
    r = tf.image.resize_images(r, [scale_size, scale_size],
                               method=tf.image.ResizeMethod.AREA)

    offset = tf.cast(tf.floor(tf.random_uniform([2], 0,
                                                scale_size - crop_size + 1,
                                                seed=seed)),
                     dtype=tf.int32)
    if scale_size > crop_size:
        r = tf.image.crop_to_bounding_box(r, offset[0], offset[1],
                                          crop_size, crop_size)
    elif scale_size < crop_size:
        raise Exception("scale size cannot be less than crop size")
    return r


def convert(image, size):
    """Resize image to size if given and convert to unit8."""
    if size:
        image = tf.image.resize_images(image, size=size,
                                       method=tf.image.ResizeMethod.BICUBIC)
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)