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
r"""Downloads and converts a particular dataset.

Usage:
```shell

$ python download_and_convert_data.py \
    --dataset_name=mnist \
    --dataset_dir=/tmp/mnist

$ python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=/tmp/cifar10

$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/flowers
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import convert_standard_dataset

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string(
    'input_dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_string(
    'output_dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_integer(
    'num_of_shards',
    5,
    'The number of shards you need to be used for TFRecord conversion')

tf.app.flags.DEFINE_integer(
    'num_of_threads',
    5,
    'The num of threads to be used for preprocessing and conversion')


def main(_):
  if not FLAGS.input_dataset_dir:
    raise ValueError('You must supply the dataset directory with --input_dataset_dir')
  if not FLAGS.output_dataset_dir:
    raise ValueError('You must supply the dataset directory with --output_dataset_dir')

  convert_standard_dataset.run(FLAGS.input_dataset_dir, FLAGS.output_dataset_dir, FLAGS.num_of_shards, FLAGS.num_of_threads)

if __name__ == '__main__':
  tf.app.run()
