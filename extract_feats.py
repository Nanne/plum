from keras.applications import VGG19
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
import numpy as np
import skimage
import skimage.transform
from skimage import io
import pickle as pkl
import h5py
import pandas as pd
from scipy.misc import imread
import os
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('img_root', '/roaming/public_datasets/MS-COCO/images/val2014/',
                           'Location where original images are stored')
tf.app.flags.DEFINE_string('record_path', '/data/cocotest.tfrecords',
                           'Directory to write the converted result to')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def preprocess_input(x):
    """Zero-center by mean pixel from ImageNet."""
    r, g, b = np.mean(x, axis=(0,1))
    x[:, :, 0] -= r
    x[:, :, 1] -= g
    x[:, :, 2] -= b
    return x


def crop_image(image_path, target_height=224, target_width=224):
    """Reshape image shorter and crop, keep aspect ratio."""
    image = skimage.img_as_float(io.imread(image_path)).astype("float32")
    # print image
    if len(image.shape) == 2:
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))
    height, width, rgb = image.shape

    if width == height:
        resized_image = skimage.transform.resize(image,
                                                 (target_height,
                                                  target_width))
    elif height < width:
        w = int(width * float(target_height)/height)
        resized_image = skimage.transform.resize(image, (target_height, w))
        crop_length = int((w - target_width) / 2)
        resized_image = resized_image[:, crop_length:crop_length+target_width]

    else:
        h = int(height * float(target_width) / width)
        resized_image = skimage.transform.resize(image, (h, target_width))
        crop_length = int((h - target_height) / 2)
        resized_image = resized_image[crop_length:crop_length+target_height, :]
    # 'RGB'->'BGR'
    if bgr:
        resized_image = resized_image[:, :, ::-1]
    return skimage.img_as_ubyte(resized_image)

def write_to(writer, image, id, path, tensors):
    """
    Write VGG19 filter responses to tfrecords.
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': _int64_feature(id),
        'path': _bytes_feature(path),
        'block2_conv2': _bytes_feature(tensors[0].tostring()),
        'block3_conv4': _bytes_feature(tensors[1].tostring()),
        'block4_conv4': _bytes_feature(tensors[2].tostring()),
        'block5_conv4': _bytes_feature(tensors[3].tostring()),
        'block2_conv2_size': _int64_features(list(tensors[0].shape)),
        'block3_conv4_size': _int64_features(list(tensors[1].shape)),
        'block4_conv4_size': _int64_features(list(tensors[2].shape)),
        'block5_conv4_size': _int64_features(list(tensors[3].shape)),
        'image': _bytes_feature(image.tostring())}))
        'image_size': _int64_features(list(image.shape)),
        }))
    writer.write(example.SerializeToString())


# Path to the visual sentiment data set
global_start = time.time()
img_files = os.listdir(FLAGS.img_root)
num_imgs = len(img_files)
# Path for the database


batch_size = 1000
bgr = True
# tensorflow (tf): b x h x w x c
b = (batch_size, 224, 224, 3)

print "Initializing model"
# VGG19 until the last fully connected layer
base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
out_layers = [base_model.get_layer('block2_conv2').output,
              base_model.get_layer('block3_conv4').output,
              base_model.get_layer('block4_conv4').output,
              base_model.get_layer('block5_conv4').output]

model = Model(input=base_model.input, output=out_layers)
writer = tf.python_io.TFRecordWriter(FLAGS.record_path)

with open(FLAGS.record_path + ".json") as f:
    json.dump({"count": num_imgs)

iters = num_imgs / batch_size
img_batch = np.empty(b)
paths = []
c = -1
print "iters", iters, "batchsize", batch_size, "num_imgs", num_imgs
start = time.time()
for i in range(0, num_imgs):
    # Store batch of activations, reset empty batch
    c += 1
    if i % batch_size == 0 and i != 0 and i != iters*batch_size:
        print "resetting", i, (i-batch_size), c
        features = model.predict(img_batch)
        #write_to(writer, i, )
        # feature_set[(i-batch_size):i] = features
        for j in range(0, len(features[0])):
            tensors = [features[0][j], features[1][j],
                       features[2][j], features[3][j]]
            write_to(writer, img_batch[j], i, paths[j], tensors)

        img_batch = np.empty(b)
        c = 0
        paths = []
        print 'Batch processed in:', time.time() - start
        start = time.time()
    # Last batch
    elif i == iters * batch_size:
        print i, num_imgs
        img_batch = np.empty((num_imgs%batch_size, 224, 224, 3))
        c = 0
        paths = []

    print i, '\r',
    path = FLAGS.img_root + img_files[i]
    cropped = crop_image(path).astype("float32")
    standardized = preprocess_input(cropped)
    img_batch[c] = cropped
    paths.append(path)


# Extracting from last batch
features = model.predict(img_batch)
for j in range(0, len(features[0])):
    tensors = [features[0][j], features[1][j],
               features[2][j], features[3][j]]
    print paths[j]
    write_to(writer, img_batch[j], i, paths[j], tensors)
