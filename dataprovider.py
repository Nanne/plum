import tensorflow as tf
import glob
import os
import math
import random
from util import to_namedtuple
import json

FLAGS = tf.app.flags.FLAGS  # parse config

if FLAGS.dataset == "SALICON":
    from datareaders.SALICON import read_record, convert
if FLAGS.dataset == "SALICON_VGG":
    from datareaders.salicon_vggfeatures import read_record, deprocess_input, deprocess_output
elif FLAGS.dataset == "mscoco_objects":
    from datareaders.mscoco_objects import read_record, deprocess_input, deprocess_output
elif FLAGS.dataset == "vggfeatures":
    from datareaders.vggfeatures import read_record, deprocess_input, deprocess_output
else:
    raise ValueError('Unknown dataset option: ' + FLAGS.dataset)

def getsize(filename):
    jsfile = filename + ".json"
    if tf.gfile.Exists(jsfile):
        with open(jsfile, 'r') as f:
            N = json.load(f)['count']
    else:
        N = 0
        print "Obtaining sample count:"
        for record in tf.python_io.tf_record_iterator(filename):
            N += 1
            print N, '\r',
        with open(jsfile, 'w') as f:
            f.write(json.dumps({'count': N}))
    return N

def load_records():
    """Imports a read_record function from the module corresponding to the dataset
    and reads records, in batches, from the dataset."""
    shuffle = FLAGS.mode == "train"
    records_file = FLAGS.input_dir
    e = {} # container for examples
    if FLAGS.aux and not FLAGS.num_classes:
        raise Exception('If using auxiliary classifier give --num_classes')

    if not os.path.exists(records_file):
        raise Exception("Path to record doesn't exist", records_file)
    with tf.name_scope("load_images"):
        filename_queue = tf.train.string_input_producer([records_file])
        tensors = read_record(filename_queue)

    # Need to loop through all records and count if sample count not given
    if not FLAGS.num_samples:
        num_samples = getsize(records_file)
    else:
        num_samples = FLAGS.num_samples

    if shuffle:
        batch = tf.train.shuffle_batch(tensors,
                                       batch_size=FLAGS.batch_size,
                                       capacity=FLAGS.batch_size*8,
                                       min_after_dequeue=FLAGS.batch_size*4,
                                       num_threads=16)
    else:
        batch = tf.train.batch(tensors, batch_size=FLAGS.batch_size)

    batch = list(batch)
    steps_per_epoch = int(math.ceil(num_samples / FLAGS.batch_size))
    e["steps_per_epoch"], e["count"]  = steps_per_epoch, num_samples

    paths_batch = batch.pop(0)
    e["paths"] = paths_batch

    if FLAGS.pretrained:
        e["inputs"] = []
        for _ in range(4):
            e["inputs"].append(batch.pop(0))
    else:
        inputs_batch = batch.pop(0)
        e["inputs"] = inputs_batch

    if FLAGS.decoder:
        targets_batch = batch.pop(0)
        e["targets"] = targets_batch
    else:
        e['targets'] = None
    if FLAGS.aux:
        aux_targets_batch = batch.pop(0)
        e["aux"] = aux_targets_batch
    else:
        e['aux'] = None
    if FLAGS.pretrained:
        # Is the input comes from a pretrained model, then visualise
        # the next tensor
        image_batch = batch.pop(0)
        e["images"] = image_batch
    else:
        # otherwise we visualise the inputs
        e["images"] = e["inputs"]

    if FLAGS.discriminator:
        e['discriminator_inputs'] = e['images']
    else:
        e['discriminator_inputs'] = None

    e["deprocess_input"] = deprocess_input
    e["deprocess_output"] = deprocess_output


    examples = to_namedtuple(e, "Examples")
    return examples
