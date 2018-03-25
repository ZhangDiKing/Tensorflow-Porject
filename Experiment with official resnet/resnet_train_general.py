#Most of the code is the same as the cifat10_main.py in tensorflow/offical/resnet
#I modified to make the code suitable for my own dataset
"""Runs a ResNet model on my own dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import resnet_model
from official.resnet import resnet_run_loop

import win_unicode_console

win_unicode_console.enable()
#the default height and width, no need for change for resnet
_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS 
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1

#this setting need to be changed for different classes for different dataset
_NUM_CLASSES = 60

#change the nums of images to your own choice
_NUM_IMAGES = {
    'train': 28736,
    'validation': 9673,
}


###############################################################################
# Data processing
###############################################################################

#for your own model, change here to make difference
def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  cwd = os.getcwd()
  if is_training:
      return cwd + data_dir + "train.tfrecords"
  else:
      return cwd + data_dir + "test.tfrecords"
  
def parse_record(raw_record, is_training):
  """Parse image and label from a raw record."""
  
  features = tf.parse_single_example(
    raw_record,
    features = {
        'label':tf.FixedLenFeature([],tf.int64),
        'image':tf.FixedLenFeature([],tf.string)
    }
  )
  image = tf.decode_raw(features['image'],tf.uint8)
  image = tf.reshape(image,[_HEIGHT, _WIDTH, _NUM_CHANNELS])
  
  label = tf.cast(features['label'], tf.int32)
  #for me, the labels in the dataset is _NUM_IMAGES*1
  #if your labels are _NUM_IMAGES*_NUM_CLASSES, comment next line
  label = tf.one_hot(label, _NUM_CLASSES)
  
  #keep the same preprocessing process
  image = preprocess_image(image, is_training)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    
  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             num_parallel_calls=1, multi_gpu=False):
  
  """Input_fn using the tf.data input pipeline for dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel.
      This can be optimized per data set but for generally homogeneous data
      sets, should be approximately the number of available CPU cores.
    multi_gpu: Whether this is run multi-GPU. Note that this is only required
      currently to handle the batch leftovers, and can be removed
      when that is handled directly by Estimator.

  Returns:
    A dataset that can be used for iteration.
  """
  filename = get_filenames(is_training, data_dir)
  dataset = tf.data.TFRecordDataset(filename)

  num_images = is_training and _NUM_IMAGES['train'] or _NUM_IMAGES['validation']

  return resnet_run_loop.process_record_dataset(
      dataset, is_training, batch_size, _NUM_IMAGES['train'],
      parse_record, num_epochs, num_parallel_calls,
      examples_per_epoch=num_images, multi_gpu=multi_gpu)


def get_synth_input_fn():
  return resnet_run_loop.get_synth_input_fn(
      _HEIGHT, _WIDTH, _NUM_CHANNELS, _NUM_CLASSES)


###############################################################################
# Running the model
###############################################################################
class MyModel(resnet_model.Model):
  """Model class with appropriate defaults for CIFAR-10 data."""

  def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
               version=resnet_model.DEFAULT_VERSION):
    """These are the parameters that work for CIFAR-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      version: Integer representing which version of the ResNet network to use.
        See README for details. Valid values: [1, 2]

    Raises:
      ValueError: if invalid resnet_size is chosen
    """
    if resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    super(MyModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=False,
        num_classes=num_classes,
        num_filters=16,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        second_pool_size=8,
        second_pool_stride=1,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        final_size=64,
        version=version,
        data_format=data_format)


def my_model_fn(features, labels, mode, params):
  """Model function for your own dataset"""
  features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=128,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[20, 40, 60],
      decay_rates=[0.1, 0.05, 0.01, 0.001])

  # We use a weight decay of 0.0002, which performs better
  # than the 0.0001 that was originally suggested.
  weight_decay = 2e-4

  def loss_filter_fn(_):
    return True

  return resnet_run_loop.resnet_model_fn(features, labels, mode, MyModel,
                                         resnet_size=params['resnet_size'],
                                         weight_decay=weight_decay,
                                         learning_rate_fn=learning_rate_fn,
                                         momentum=0.9,
                                         data_format=params['data_format'],
                                         version=params['version'],
                                         loss_filter_fn=loss_filter_fn,
                                         multi_gpu=params['multi_gpu'])


def main(argv):
  parser = resnet_run_loop.ResnetArgParser()
  # Set defaults that are reasonable for this model.
  # on gtx 960m, 21 s for 100 steps for batch_size=128, w=32, h=32, ch=3, resnet_size=32
  cwd = os.getcwd()
  parser.set_defaults(data_dir='./data_tfrecords/',
                      model_dir= cwd + '/trained_model/',
                      resnet_size=32,
                      train_epochs=80,
                      epochs_between_evals=2,
                      batch_size=128)

  flags = parser.parse_args(args=argv[1:])

  input_function = flags.use_synthetic_data and get_synth_input_fn() or input_fn
  tf.reset_default_graph()
  resnet_run_loop.resnet_main(flags, my_model_fn, input_function)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)

