import tensorflow as tf
import json
import os


tf.app.flags.DEFINE_string("input_dir", '', "path to folder containing images")
tf.app.flags.DEFINE_boolean("records", True,
                            "Use TFRecords. If true '--input_dir' \
                             must be a path to the record")
tf.app.flags.DEFINE_integer("num_samples", None,
                            "When using TFrecords the number of samples have to \
                             be calculated at runtime. Providing a value for \
                             --num_samples bypasses this computation.")
tf.app.flags.DEFINE_string("mode", 'train', "Choose 'train' 'test' 'export'")
tf.app.flags.DEFINE_string("output_dir", None, "where to put output files")
tf.app.flags.DEFINE_string("checkpoint", None,
                           "directory with checkpoint to resume training \
                            from or use for testing")

tf.app.flags.DEFINE_boolean("decoder", True,
                            'Attach top-half (decoder) of U-net. \
                             If True trains with --content_loss'  )
tf.app.flags.DEFINE_boolean("aux", False,
                            "Do auxiliary classification with encoder \
                             If True trains with binary cross-entropy.")
tf.app.flags.DEFINE_boolean("discriminator", True,
                            "Add or remove discriminator on U-net. \
                             If True train with GAN loss on top \
                             of content loss.")

tf.app.flags.DEFINE_string("architecture", 'unet', "Choose 'unet' 'vgg19'")
tf.app.flags.DEFINE_string("dataset", 'SALICON', "Choose 'SALICON' 'mscoco_objects'")

tf.app.flags.DEFINE_integer("ngf", 64,
                            "number of generator filters in first conv layer \
                             the decoder.")
tf.app.flags.DEFINE_integer("--ndf", 64,
                            "number of discriminator filters \
                             in first conv layer of the discriminator")

tf.app.flags.DEFINE_string("content_loss", "bat",
                           "Choose 'bat', 'L1' or 'both' \
                            for Bhattaccarya, L1 or interpolated loss")
tf.app.flags.DEFINE_float("content_weight", 100.0,
                          "weight on content term for generator gradient")
tf.app.flags.DEFINE_float("aux_weight", 10.0,
                          "weight on aux term for generator gradient")
tf.app.flags.DEFINE_integer("num_classes", None,
                            "Number of classes for auxiliary prediction")
tf.app.flags.DEFINE_float("gan_weight", 1.0,
                          "weight on GAN term for generator gradient")


tf.app.flags.DEFINE_integer("max_epochs", 0, "number of training epochs")
tf.app.flags.DEFINE_integer("summary_freq", 100,
                            "update summaries every summary_freq steps")
tf.app.flags.DEFINE_integer("progress_freq", 50,
                            "display progress every progress_freq steps")
# to get tracing working on GPU, LD_LIBRARY_PATH may need to be modified:
# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64
tf.app.flags.DEFINE_integer("trace_freq", 0,
                            "trace execution every trace_freq steps")
tf.app.flags.DEFINE_integer("display_freq", 0,
                            "write current training images \
                             every display_freq steps")
tf.app.flags.DEFINE_integer("save_freq", 5000,
                            "save model every save_freq steps, 0 to disable")

tf.app.flags.DEFINE_float("aspect_ratio", 1.0,
                          "aspect ratio of output images (width/height)")
tf.app.flags.DEFINE_integer("batch_size", 1, "number of images in batch")
tf.app.flags.DEFINE_string("which_direction", "AtoB", "Choose 'AtoB', 'BtoA'")
tf.app.flags.DEFINE_integer("scale_size", 286,
                            "scale images to this size before \
                             cropping to 256x256")
tf.app.flags.DEFINE_integer("upsample_h", None, "Upsample image to H x W")
tf.app.flags.DEFINE_integer("upsample_w", None, "Upsample image to H x W")
tf.app.flags.DEFINE_boolean("salicon", False,
                            "Load settings from config file \
                             salicon.cfg for convenience")

tf.app.flags.DEFINE_boolean("flip", True, "flip images horizontally")
tf.app.flags.DEFINE_boolean("instancenorm", False,
                            "Do instancenorm instead of batchnorm")
tf.app.flags.DEFINE_boolean("no_flip", False,
                            "don't flip images horizontally")
tf.app.flags.DEFINE_float("lr", 0.0002, "initial learning rate for adam")
tf.app.flags.DEFINE_float("beta1", 0.5, "momentum term of adam")
tf.app.flags.DEFINE_integer("seed", 1860795210, "Random seed")

"""Restore options from checkpoint/options.json."""
restore_flags = {"which_direction", "ngf", "ndf",
           "gan_weight", "content_weight", "lr", "beta1",
           "trace_freq", "summary_freq", "aux", "aux_weight",
           "num_classes", "discriminator", "instancenorm",
           "content_loss", "decoder"}
