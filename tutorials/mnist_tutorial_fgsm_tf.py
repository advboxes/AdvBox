# Copyright 2017 - 2018 Baidu Inc.
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
"""
FGSM tutorial on mnist using advbox tool.
FGSM method is non-targeted attack while FGSMT is targeted attack.
"""
import sys
sys.path.append("..")
import logging
logging.basicConfig(level=logging.INFO,format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger=logging.getLogger(__name__)

#import matplotlib.pyplot as plt
import numpy as np

from advbox.adversary import Adversary
from advbox.attacks.gradient_method import FGSM
from advbox.models.tensorflow import TensorflowModel
from tutorials.mnist_model_tf import mnist_cnn_model
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main(dirname):
    """
    Advbox demo which demonstrate how to use advbox.
    """
    TOTAL_NUM = 500

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    x = tf.placeholder(tf.float32, [None, 784])

    y_ = tf.placeholder(tf.int64, [None])

    logits, keep_prob = mnist_cnn_model(x)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy( labels=y_, logits=logits)
    cross_entropy = tf.reduce_mean(cross_entropy)


    BATCH_SIZE = 1

    def create_graph(dirname):
        with tf.gfile.FastGFile(dirname, 'rb') as f:
            #graph_def = tf.GraphDef()
            graph_def=sess.graph_def
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')


    with tf.Session() as sess:
        # 加载pb文件
        create_graph(dirname)

        # advbox demo
        m = TensorflowModel(
            sess,
            x,
            cross_entropy,
            logits, (-1, 1),
            channel_axis=1)
        attack = FGSM(m)
        attack_config = {"epsilons": 0.3}

        # use test data to generate adversarial examples
        total_count = 0
        fooling_count = 0

        for _ in range(10000):
            data = mnist.test.next_batch(BATCH_SIZE, shuffle=False)

            total_count += 1
            (x,y)=data

            y=y[0]

            #print(x.shape)
            #print(y.shape)

            adversary = Adversary(x,y)

            # FGSM non-targeted attack
            adversary = attack(adversary, **attack_config)


            if adversary.is_successful():
                fooling_count += 1
                print(
                    'attack success, original_label=%d, adversarial_label=%d, count=%d'
                    % (y, adversary.adversarial_label, total_count))

            else:
                print('attack failed, original_label=%d, count=%d' %
                      (y, total_count))

            if total_count >= TOTAL_NUM:
                print(
                    "[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                    % (fooling_count, total_count,
                       float(fooling_count) / total_count))
                break
        print("fgsm attack done")


if __name__ == '__main__':
    main(dirname="./mnist-tf/cnn.pb")
