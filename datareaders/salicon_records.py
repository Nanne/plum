import re
import sys
from collections import Counter
import numpy as np
import spacy
import json
import tensorflow as tf
from scipy import misc
sys.path.append("/roaming/public_datasets/MS-COCO/coco/PythonAPI/pycocotools/")
sys.path.append('/roaming/public_datasets/SALICON/PythonAPI/')
sys.path.append("..")
from pycocotools import coco
from salicon.salicon import SALICON
import os
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.models import Model
from keras.preprocessing import image
import skimage
import skimage.transform
from skimage import io


SALICON_ROOT = '/roaming/public_datasets/SALICON/'
COCO_ROOT = "/roaming/public_datasets/MS-COCO/"
objsFile_train = COCO_ROOT + 'annotations/annotations/instances_train2014.json'
capsFile_train = COCO_ROOT + 'annotations/captions_train2014.json'
objsFile_val = COCO_ROOT + 'annotations/annotations/instances_val2014.json'
capsFile_val = COCO_ROOT + 'annotations/captions_val2014.json'
# train_imgs = '/roaming/public_datasets/SALICON/small_trial_2/combined/train/'
# val_imgs = '/roaming/public_datasets/SALICON/small_trial_2/combined/val/'
train_imgs = COCO_ROOT+"images/train2014/"
val_imgs = COCO_ROOT+"images/val2014/"
salicon_train = SALICON_ROOT + 'annotations/fixations_train2015r1.json'
salicon_val = SALICON_ROOT + 'annotations/fixations_val2015r1.json'
vgg = True

nlp = spacy.load('en')
MAX_LENGTH = 60  # chop of captions, caption[:MAX_LENGTH]
N_OBJECTS = 90

# Data helper functions
def repeat(L, n):
    """Repeat all elements in L n times."""
    return [item for item in L for i in range(n)]


def get_imgid(f):
    """Given MS-COCO image filename return cocoid."""
    return int(f.split('/')[-1].split('.')[0].split('_')[-1])


def clean(s):
    """Remove non-alphabetic characters from and lowercase string."""
    return re.sub('[^A-Za-z0-9 ]+', '', s).lower()


def load_caps(cocoid):
    """Load caption given cocoid. Only return first 5."""
    capIds = coco_cap.getAnnIds(imgIds=cocoid)
    caps = coco_cap.loadAnns(capIds)[:5]
    caps = [clean(x['caption']) for x in caps]
    return caps


def load_objs(cocoid):
    """Return object names given cocoid."""
    catIds = coco_obj.getCatIds(catNms=coco_obj.cats)
    objIds = coco_obj.getAnnIds(imgIds=cocoid, catIds=catIds)
    objs = coco_obj.loadAnns(objIds)
    objs = [x['category_id'] for x in objs]
    obj_names = map(lambda x: coco_obj.cats[x]['name'], objs)
    return set(obj_names)


def extract_nouns(caps):
    """Return the set of noun-lemmas in a list of captions."""
    nouns = []
    for c, sent in enumerate(caps):
        doc = nlp(unicode(sent))
        nouns += [x.lemma_ for x in doc if x.pos_ == "NOUN"]
    return set(nouns)


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
    resized_image = resized_image[:, :, ::-1]
    return resized_image


def encode_caps(captions):
    """Encode captions based on dictionary. Returns 5 x MAX_LENGTH matrix."""
    C = np.empty((5, MAX_LENGTH))   # container for captions vectors
    l = []                          # container for captions lengths
    for i, cap in enumerate(captions):
        cap = cap.split()
        l.append(min(len(cap), MAX_LENGTH))
        cap = cap + ["STOP"] * max(0, MAX_LENGTH - len(cap))
        # Encode UNK words as 1
        cap = [caps_dict.get(x, 1) for x in cap]
        C[i] = cap
    return C, l

# TFRecrods helper functions


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def write_to(writer, cocoid, img_paths, objects):
    """
    Conver data tf.train.Example and SerializeToString.

    writer: tf.python_io.TFRecordWriter
    cocoid: integer, cocoid
    filename: path to stiched img, half img COCO image half SALICON salmap
    captions: matrix of captions mapped to ints, size MAX_length
    nouns: nouns mapped to ints
    objects: objects given cocoid mapped to ints
    caption_lengths: list of ints, length of each caption before padding
    """
    filename = coco_obj.loadImgs(cocoid)[0]['file_name'].split(".")[0] + ".png"
    path = img_paths + filename
    image_raw = misc.imread(path)
    h, w, d = image_raw.shape
    objs_vector = np.zeros(N_OBJECTS, dtype="int")
    objs_vector[objects] = 1
    if image_raw.shape != (480, 1280, 3):
        print image_raw.shape
        print filename
        sys.exit()
    example = tf.train.Example(features=tf.train.Features(feature={
        'cocoid': _int64_feature(int(cocoid)),
        'path': _bytes_feature(str(filename)),
        'height': _int64_feature(int(h)),
        'width': _int64_feature(int(w)),
        'depth': _int64_feature(int(d)),
        'image_raw': _bytes_feature(image_raw.tostring()),
        'objects':  _bytes_feature(objs_vector.tostring()),
        }))

    writer.write(example.SerializeToString())


def write_to_AB(writer, cocoid, img_paths, objects, cnn=None):
    """
    Conver data tf.train.Example and SerializeToString.

    writer: tf.python_io.TFRecordWriter
    cocoid: integer, cocoid
    filename: path to stiched img, half img COCO image half SALICON salmap
    captions: matrix of captions mapped to ints, size MAX_length
    nouns: nouns mapped to ints
    objects: objects given cocoid mapped to ints
    caption_lengths: list of ints, length of each caption before padding

    Write Image and Salmap as 2 entries rather than stiched.
    """
    filename = coco_obj.loadImgs(cocoid)[0]['file_name']
    path = img_paths + filename
    image_raw = misc.imread(path)
    if not vgg:
        if image_raw.shape != (480, 640, 3):
            print "Wrong image shape", image_raw.shape
            print filename
            if image_raw.shape == (480, 640):
                image_raw = np.tile(image_raw[:, :, np.newaxis], (1, 1, 3))
            else:
                sys.exit()
    annIds = salicon_handler.getAnnIds(cocoid)
    anns = salicon_handler.loadAnns(annIds)
    fixmap = salicon_handler.buildFixMap(anns).astype("float32")
    fixmap = np.expand_dims(fixmap, axis=2)
    if fixmap.shape != (480, 640, 1):
        print "Wrong fixation map shape", fixmap.shape
        print filename
        sys.exit()
    fixmap = skimage.transform.resize(fixmap, (224, 224))
    objs_vector = np.zeros(N_OBJECTS, dtype="int")
    objs_vector[objects] = 1
    if vgg:
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        tensors = cnn.predict(x)
        example = tf.train.Example(features=tf.train.Features(feature={
            'cocoid': _int64_feature(int(cocoid)),
            'path': _bytes_feature(str(filename)),
            'height': _int64_feature(224),
            'width': _int64_feature(224),
            'depth': _int64_feature(3),
            'image_orig': _bytes_feature(img.tobytes()),
            'image': _bytes_feature(x.tostring()),
            'fixmap': _bytes_feature(fixmap.tostring()),
            'objects':  _bytes_feature(objs_vector.tostring()),
            'block2_conv2': _bytes_feature(tensors[0].tostring()),
            'block3_conv4': _bytes_feature(tensors[1].tostring()),
            'block4_conv4': _bytes_feature(tensors[2].tostring()),
            'block5_conv4': _bytes_feature(tensors[3].tostring()),
            'block2_conv2_size': _int64_features(list(tensors[0].shape)),
            'block3_conv4_size': _int64_features(list(tensors[1].shape)),
            'block4_conv4_size': _int64_features(list(tensors[2].shape)),
            'block5_conv4_size': _int64_features(list(tensors[3].shape)),
            }))
    else:
        example = tf.train.Example(features=tf.train.Features(feature={
            'cocoid': _int64_feature(int(cocoid)),
            'path': _bytes_feature(str(filename)),
            'height': _int64_feature(int(h)),
            'width': _int64_feature(int(w)),
            'depth': _int64_feature(int(d)),
            'image_raw': _bytes_feature(image_raw.tostring()),
            'fixmap': _bytes_feature(fixmap.tostring()),
            'objects':  _bytes_feature(objs_vector.tostring()),
            }))
    writer.write(example.SerializeToString())

def write_tfrecords(writer, ids, coco_obj, img_paths, cnn):
    """Write TfRecords for a given split."""
    for c, i in enumerate(ids):
        print c, '\r',
        objs = load_objs(i)
        # caps = load_caps(i)
        # nouns = extract_nouns(caps)
        # ids for unique objs
        # Use 0 based indexing
        int_objs = list(set([objs_dict[x] - 1 for x in objs]))
        # ids for unique nouns, encode UNK nouns as 0
        # int_nouns = set([nouns_dict.get(x, 0) for x in nouns])
        # int_caps, lengths = encode_caps(caps)            # captions matrix, L
        write_to_AB(writer, i, img_paths, int_objs, cnn)


def select_caption(captions):
    """Choose a row from the captions matrix."""
    w = tf.shape(captions)[1]
    caption = tf.random_crop(captions, [1, w])
    return caption


if __name__ == "__main__":
    # TODO USE SALICON IDS NOT THE COCO IDS !!!!!
    out_dir = "/data/SALICON"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if vgg:
        base_model = VGG19(weights='imagenet', include_top=False,
                           pooling='avg')
        out_layers = [base_model.get_layer('block2_conv2').output,
                      base_model.get_layer('block3_conv4').output,
                      base_model.get_layer('block4_conv4').output,
                      base_model.get_layer('block5_conv4').output]

        cnn = Model(input=base_model.input, output=out_layers)
    else:
        cnn = None

    coco_obj = coco.COCO(objsFile_train)  # COCO object annotations handler
    coco_cap = coco.COCO(capsFile_train)  # COCO caption annotations handler
    salicon_handler = SALICON(salicon_train)
    ids = salicon_handler.getImgIds()
    print "Creating class2id mappers for objects"
    objs_dict = {v['name']: k for k, v in coco_obj.cats.items()}
    print "Creating class2id mappers for tokens in the captions"
    caps = [y for x in ids for y in load_caps(x)]       # all captions
    tokens = [y for x in caps for y in x.split()]       # all tokens
    word_counts = Counter(tokens)                       # token counts
    object_counts = Counter([y for x in ids for y in load_objs(x)])
    print object_counts
    caps_dict = dict(zip(word_counts.keys(), range(2, len(word_counts))))
    print "Creating class2id mappers for nouns"
    nouns = extract_nouns(caps)      # all unique nouns
    nouns_dict = dict(zip(nouns, range(1, len(nouns))))
    caps_dict["STOP"] = 0
    caps_dict["UNK"] = 1
    nouns_dict["UNK"] = 0
    print "Writing mappers to json"
    json.dump(objs_dict, open(out_dir + "/object_mapper.json",  'w'))
    json.dump(nouns_dict, open(out_dir + "/noun_mapper.json", 'w'))
    json.dump(caps_dict, open(out_dir + "/caption_mapper.json", 'w'))
    json.dump(word_counts, open(out_dir + "/word_counts.json", 'w'))
    json.dump(object_counts, open(out_dir + "/object_counts.json", 'w'))
    objs_dict = json.load(open(out_dir + "/object_mapper.json"))
    nouns_dict = json.load(open(out_dir + "/noun_mapper.json"))
    caps_dict = json.load(open(out_dir + "/caption_mapper.json"))

    print "Writing records for training set"
    writer = tf.python_io.TFRecordWriter(out_dir + "/train_AB_vgg.tfrecords")
    write_tfrecords(writer, ids, coco_obj, train_imgs, cnn)
    writer.close()
    print "Done"

    print "Writing records for validation set"
    coco_obj = coco.COCO(objsFile_val)  # COCO object annotations handler
    coco_cap = coco.COCO(capsFile_val)  # COCO caption annotations handler
    salicon_handler = SALICON(salicon_val)  # SALICON handler
    ids = salicon_handler.getImgIds()       # ids of val set

    writer = tf.python_io.TFRecordWriter(out_dir + "/val_AB_vgg.tfrecords")
    write_tfrecords(writer, ids, coco_obj, val_imgs, cnn)
    writer.close()
