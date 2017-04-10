from keras.applications import VGG19
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image as keras_image
import numpy as np
import skimage.transform
import h5py, json
import tensorflow as tf
import time, os

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def crop_image(image, target_height=224, target_width=224):
    """Reshape image shorter and crop, keep aspect ratio."""
    width, height = image.size

    if width == height:
        resized_image = image.resize((target_width,
                                      target_height))

    elif height < width:
        w = int(width * float(target_height)/height)
        resized_image = image.resize((w, target_height))

        crop_length = int((w - target_width) / 2)
        resized_image = resized_image.crop((crop_length, 
                                           0,
                                           crop_length+target_width,
                                           target_height))
    else:
        h = int(height * float(target_width) / width)
        resized_image = image.resize((target_width, h))

        crop_length = int((h - target_height) / 2)
        resized_image = resized_image.crop((0,
                                           crop_length, 
                                           target_width,
                                           crop_length+target_height))
    return resized_image

def write_to(writer, id, image, path, tensors):
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
        'image': _bytes_feature(image.tobytes()),
        'image_size': _int64_features(list(image.size) + [3])
        }))
    writer.write(example.SerializeToString())

if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('img_root', '/roaming/public_datasets/MS-COCO/images/val2014/',
                               'Location where original images are stored')
    tf.app.flags.DEFINE_string('record_path', '/data/cocotest.tfrecords',
                               'Directory to write the converted result to')

    # Path to the visual sentiment data set
    img_files = os.listdir(FLAGS.img_root)
    num_imgs = len(img_files)

    print "Initializing model"
    # VGG19 until the last fully connected layer
    base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
    out_layers = [base_model.get_layer('block2_conv2').output,
                  base_model.get_layer('block3_conv4').output,
                  base_model.get_layer('block4_conv4').output,
                  base_model.get_layer('block5_conv4').output]

    model = Model(input=base_model.input, output=out_layers)
    writer = tf.python_io.TFRecordWriter(FLAGS.record_path)

    with open(FLAGS.record_path + ".json", 'w') as f:
        json.dump({"count": num_imgs}, f)

    print "Writing", num_imgs, "images"
    start = time.time()
    for i in range(0, num_imgs):
        print i, '\r',
        path = os.path.join(FLAGS.img_root, img_files[i])

        image = keras_image.load_img(path)
        cropped = crop_image(image)
        standardized = keras_image.img_to_array(cropped)
        standardized = np.expand_dims(standardized, axis=0)
        standardized = preprocess_input(standardized)

        features = model.predict(standardized)

        tensors = [features[0], features[1],
               features[2], features[3]]

        write_to(writer, i, cropped, path, tensors)

    print 'Data processed in:', time.time() - start
