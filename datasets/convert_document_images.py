# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import csv
import tensorflow as tf
import cv2
from datasets import dataset_utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset to convert, one of "cifar10", "flowers", "mnist".')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')


# The number of images in the validation set.
_NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_png(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir, split_name):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  annotations_root = os.path.join(dataset_dir, 'labels')
  images_root = os.path.join(dataset_dir, 'images')
  split_file = os.path.join(annotations_root, split_name)
  file_names = []
  labels = []
  counter = 0
  with open(split_file + '.txt') as csvFile:
      read_object = csv.reader(csvFile, delimiter= ' ')
      for row in read_object:
          print(row[0])
          abs_file = os.path.join(images_root, row[0])
          # file_names.append(abs_file)
          image = cv2.imread(abs_file)
          new_name = row[0].split('.')[0]
          final_image_name = os.path.join(images_root, new_name + '.png')
          print(final_image_name)
          print(cv2.imwrite(final_image_name, image))
          file_names.append(final_image_name)
          labels.append(int(row[1]))
          counter+=1
          if counter>50:
              break
  return file_names, labels

def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'documents_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, image_names, image_labels, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  filenames = image_names
  assert split_name in ['train', 'val']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            print (filenames[i])
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_id = int(image_labels[i])

            example = dataset_utils.image_to_tfexample(
                image_data, b'png', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _get_labels_map(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  dataset_root = dataset_dir
  labels_map_file = os.path.join(dataset_root, 'label_map.txt')
  label_id = []
  label_name = []

  with open(labels_map_file) as csvFile:
      read_object = csv.reader(csvFile, delimiter= ' ')
      for row in read_object:
          label_id.append(row[0])
          label_name.append(row[1])
  return label_name, label_id


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  tf_record_directory = dataset_dir + '/tf_record'
  if not tf.gfile.Exists(tf_record_directory):
    tf.gfile.MakeDirs(tf_record_directory)

  # if _dataset_exists(tf_record_directory):
  #   print('Dataset files already exist. Exiting without re-creating them.')
  #   return

  # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  image_names, image_labels = _get_filenames_and_classes(dataset_dir, split_name='train')
  class_names, class_id = _get_labels_map(dataset_dir)
  print(class_names)
  class_names_to_ids = dict(zip(class_names, class_id))
  _convert_dataset(split_name = 'train', dataset_dir = tf_record_directory, image_names = image_names, image_labels = image_labels)

  image_names, image_labels = _get_filenames_and_classes(dataset_dir, split_name='val')
  class_names, class_id = _get_labels_map(dataset_dir)
  print(class_names)
  class_names_to_ids = dict(zip(class_names, class_id))
  _convert_dataset(split_name = 'val', dataset_dir = tf_record_directory, image_names = image_names, image_labels = image_labels)
  # _convert_dataset('validation', validation_filenames, class_names_to_ids,
  #                  dataset_dir)
  #
  # # Finally, write the labels file:
  # labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
  #
  # _clean_up_temporary_files(dataset_dir)
  # print('\nFinished converting the Flowers dataset!')

def main(_):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  run(FLAGS.dataset_dir)

if __name__ == '__main__':
  tf.app.run()
